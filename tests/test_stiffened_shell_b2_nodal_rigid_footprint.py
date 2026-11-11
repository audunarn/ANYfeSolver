"""Reference-linear selected-node B2 footprints; no weld/drill qualification."""
from dataclasses import asdict, FrozenInstanceError, replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver._stiffened_shell_b2_nodal_footprint import (
    LAW_ID, _Incidence, compile_footprints, reference_shell_matrix,
    selected_wrenches,
)
from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint


def _definition(*, cells=2, points=False, affine=False, selected_only=False):
    nodes = {1+i: (float(i), 0., 0.) for i in range(cells+1)}
    nodes.update({cells+2+i: (float(i), 1., 0.) for i in range(cells+1)})
    shells = tuple(api.Q4PanelShell(
        f"shell-{i}", (i+1, i+2, cells+3+i, cells+2+i), 1000., .3, .15)
        for i in range(cells))
    section = api.LinearSection.isotropic(
        E=2000., nu=.25, area=.04, Iy=.00015, Iz=.0003,
        J=.0004, ky=.8, kz=.7)
    members = tuple(api.PhysicalB2Member(
        f"beam-{i}", f"source-{i}", (10+i, 11+i),
        ((float(i), .5, .2), (float(i+1), .5, .2)),
        (0., 1., 0.), "section", section) for i in range(cells))
    old_points = ((api.Q4B2Point("old-point", "shell-0", "beam-0", .5),)
                  if points else ())
    selected = shells[-1].node_ids[:3] if selected_only else shells[-1].node_ids
    group = api.B2NodalRigidFootprint(
        "right-patch", selected, 10+cells-1,
        () if selected_only else (shells[-1].node_ids,))
    total = 6*(len(nodes)+cells+1)
    fixed = set(range(6)) | set(range(6*(cells+1), 6*(cells+2)))
    fixed |= set(range(6*len(nodes), 6*len(nodes)+6))
    free = np.array([i for i in range(total) if i not in fixed])
    T = sparse.eye(total, format="csr")[:, free]
    u0 = np.zeros(total)
    if affine:
        change = sparse.eye(len(free), format="lil")
        change[0, 1] = .17
        T = (T@change.tocsr()).tocsr()
        u0[:6] = (1e-4, -2e-4, 3e-5, 2e-4, -1e-4, 3e-4)
    return api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, old_points, T, u0, (group,))


def _owner(request, definition):
    made = api.create_analysis(definition=definition)
    request.addfinalizer(made.close)
    return made


def _load(definition):
    total = definition.transformation.shape[0]
    load = np.zeros(total)
    load[-5] = 2.
    load[-4] = -.5
    load[-3] = .1
    load[-1] = .15
    return load


def _cross(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _independent(definition):
    """Separate element scatter and explicit physical node equations."""
    shell_ids = sorted(definition.shell_nodes_m)
    beam_xyz = {n: xyz for m in definition.members
                for n, xyz in zip(m.node_ids, m.reference_points_m)}
    beam_ids = sorted(beam_xyz)
    si = {n: i for i, n in enumerate(shell_ids)}
    bi = {n: len(si)+i for i, n in enumerate(beam_ids)}
    total = 6*(len(si)+len(bi))
    K = np.zeros((total, total))
    model = FEModel("independent-B2-nodal-footprint")
    for n, xyz in definition.shell_nodes_m.items():
        model.add_node(n, *xyz)
    for index, shell in enumerate(definition.shells, 1):
        material = f"material-{index}"
        model.add_material(material, shell.E, shell.nu)
        element = QualifiedE4PLShellElement(index, shell.node_ids, material,
            thickness=shell.thickness_m, reference_normal=np.array(shell.reference_normal))
        model.add_element(index, element)
        slots = [6*si[n]+j for n in shell.node_ids for j in range(6)]
        K[np.ix_(slots, slots)] += element.compute_stiffness_matrix(
            model.mesh, model.get_material(material))
    for member in definition.members:
        slots = [6*bi[n]+j for n in member.node_ids for j in range(6)]
        transform = member.local_dof_map()
        K[np.ix_(slots, slots)] += transform.T@member.local_operator().evaluate(
            np.zeros(12)).tangent@transform
    assert not definition.points  # Mixed provenance checked separately below.
    rows = []
    for group in definition.footprints:
        for n in group.shell_nodes:
            row = np.zeros((6, total))
            d = np.asarray(beam_xyz[group.beam_node])-definition.shell_nodes_m[n]
            row[:, 6*si[n]:6*si[n]+6] = -np.eye(6)
            row[:3, 6*si[n]+3:6*si[n]+6] = _cross(d)
            row[:, 6*bi[group.beam_node]:6*bi[group.beam_node]+6] = np.eye(6)
            rows.append(row)
    return K, np.vstack(rows), shell_ids, beam_ids, beam_xyz


@pytest.mark.parametrize("affine", (False, True))
def test_full_physical_kkt_wrench_and_prescribed_work(request, affine):
    definition = _definition(affine=affine)
    owner = _owner(request, definition)
    K, C, *_ = _independent(definition)
    np.testing.assert_array_equal(owner._K.toarray(), K)
    np.testing.assert_array_equal(owner._C.toarray(), C)
    T = definition.transformation.toarray(); u0 = definition.prescribed_offset
    n = T.shape[1]; r = C.shape[0]
    A = np.block([[T.T@K@T, T.T@C.T], [C@T, np.zeros((r, r))]])
    load = _load(definition)
    expected = np.linalg.solve(A, np.r_[T.T@(load-K@u0), -C@u0])
    u = T@expected[:n]+u0; lam = expected[n:]
    selection = api.B2NodalRigidFootprintSelection(("right-patch",))
    result = owner.solve_linear(load, selected_footprints=selection)
    assert result.status == "completed", result.failure_reason
    assert type(result) is api.ExperimentalB2Q4FootprintResult
    np.testing.assert_allclose(result.displacements, u, rtol=1e-8, atol=1e-12)
    expected_reaction = K@u+C.T@lam-load
    np.testing.assert_allclose(result.affine_reactions, expected_reaction, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(T.T@result.affine_reactions, 0., atol=1e-9)
    np.testing.assert_allclose(C@result.displacements, 0., atol=1e-12)
    group = result.footprint_wrenches[0]
    shell_action = np.zeros(len(u)); beam_action = np.zeros(len(u))
    for action in group.node_actions:
        item = next(x for x in owner._footprint_incidence if x.row_start == action.row_start)
        local_lam = lam[action.row_start:action.row_start+6]
        shell_expected = -C[action.row_start:action.row_start+6, item.shell_slots].T@local_lam
        np.testing.assert_allclose(action.shell_force_N+action.shell_direct_couple_Nm,
                                   shell_expected, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(action.beam_force_N+action.beam_direct_couple_Nm,
                                   -local_lam, rtol=1e-8, atol=1e-9)
        shell_action[list(item.shell_slots)] += action.shell_force_N+action.shell_direct_couple_Nm
        beam_action[list(item.beam_slots)] += action.beam_force_N+action.beam_direct_couple_Nm
    np.testing.assert_allclose(group.shell_force_N, -np.asarray(group.beam_force_N), atol=1e-9)
    np.testing.assert_allclose(group.shell_moment_Nm, -np.asarray(group.beam_moment_Nm), atol=1e-9)
    np.testing.assert_allclose(shell_action+beam_action, -C.T@lam, rtol=1e-8, atol=1e-9)
    virtual = np.linspace(-.03, .04, len(u))
    assert abs(virtual@(shell_action+beam_action)+lam@(C@virtual)) < 1e-9
    work = u@load+u0@expected_reaction
    assert abs(2*result.internal_strain_energy-work) <= 1e-8*abs(work)
    if affine:
        assert abs(u0@expected_reaction) > 0.
    assert result.point_multipliers == result.point_gaps == ()
    assert owner.solve_linear(load, recover_members={}).footprint_wrenches == ()


@pytest.mark.parametrize("mode", range(6))
def test_common_rigid_fields_zero_gap_and_element_deformation(request, mode):
    definition = _definition()
    owner = _owner(request, definition)
    K, C, shell_ids, beam_ids, beam_xyz = _independent(definition)
    a = np.zeros(3); w = np.zeros(3)
    (a if mode < 3 else w)[mode % 3] = .01
    xyz = [definition.shell_nodes_m[n] for n in shell_ids]+[beam_xyz[n] for n in beam_ids]
    rigid = np.concatenate([np.r_[a+np.cross(w, p), w] for p in xyz])
    np.testing.assert_allclose(C@rigid, 0., atol=1e-14)
    residual = K@rigid
    bound = np.abs(K)@np.abs(rigid)
    assert np.max(np.abs(residual), initial=0.) <= 1e-12*np.max(bound, initial=1.)
    np.testing.assert_allclose(owner._C@rigid, 0., atol=1e-14)


@pytest.mark.parametrize("units", (1., 1000.))
def test_nonaxis_frame_and_work_conjugate_unit_covariance(units):
    # Independent common rotation and translations; no fitted material frame.
    axis = np.array((1., 2., -1.)); axis /= np.linalg.norm(axis)
    S = _cross(axis); angle = .63
    R = np.eye(3)+np.sin(angle)*S+(1-np.cos(angle))*(S@S)
    xs = np.array((.2, -.3, .4)); xb = np.array((1.1, .5, -.2))
    column = np.diag((units,)*3+(1.,)*3)
    rotation = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    row = column@rotation
    transformed = reference_shell_matrix(units*(R@xs), units*(R@xb))
    original = reference_shell_matrix(xs, xb)
    np.testing.assert_allclose(transformed@(column@rotation), row@original, atol=1e-12*units)
    u = np.arange(6)*.013; v = np.arange(6)[::-1]*.017
    lam = np.array((2., -3., 5., .7, -.4, .9))
    new_lam = np.linalg.solve(row.T, lam)
    assert abs(new_lam@(transformed@(column@rotation@u)+column@rotation@v)
               -lam@(original@u+v)) <= 1e-12


def test_reference_rows_are_objective_pose_law_derivative_with_nonzero_multiplier():
    positions = np.array(((.2, -.3, .4), (1.1, .5, -.2)))
    frames = np.array((np.eye(3), np.eye(3)))
    lam = np.array((2., -3., 5., .7, -.4, .9))
    joint = RigidPoseJoint(positions, frames)
    evaluated = joint.evaluate(positions, frames, lam)
    C = np.column_stack((reference_shell_matrix(*positions), np.eye(6)))
    np.testing.assert_allclose(evaluated.constraint_jacobian, C, atol=1e-12)
    # Independent actual left-exponential poses. This assesses the virgin
    # linearization; it does not admit a nonlinear B2 owner.
    direction = np.linspace(-.17, .23, 12).reshape(2, 6)
    def gap(epsilon):
        made = []
        for row in direction:
            w = epsilon*row[3:]; angle = np.linalg.norm(w)
            S = _cross(w)
            made.append(np.eye(3)+np.sin(angle)/angle*S+
                        (1-np.cos(angle))/angle**2*(S@S))
        return joint.evaluate(positions+epsilon*direction[:, :3],
                              np.asarray(made), lam).constraints
    for epsilon in (1e-4, 3e-5, 1e-5):
        derivative = (gap(epsilon)-gap(-epsilon))/(2*epsilon)
        np.testing.assert_allclose(derivative, C@direction.ravel(), rtol=1e-6, atol=1e-9)
        assert abs(lam@derivative-lam@(C@direction.ravel())) < 1e-6


@pytest.mark.parametrize("member_index", (0, 1))
@pytest.mark.parametrize("reverse", (False, True))
def test_shared_endpoint_is_actual_physical_node_trace(member_index, reverse):
    definition = _definition()
    member = definition.members[member_index]
    if reverse:
        member = replace(member, node_ids=member.node_ids[::-1],
                         reference_points_m=member.reference_points_m[::-1])
    index = member.node_ids.index(11)
    trace = member.trace_global(np.zeros(12), stations=(float(index),))
    expected = np.zeros((6, 12)); expected[:, 6*index:6*index+6] = np.eye(6)
    np.testing.assert_allclose(trace.jacobian[0], expected, atol=1e-12)
    np.testing.assert_allclose(trace.reference_points_m[0], member.reference_points_m[index], atol=1e-12)


def test_point_rows_first_and_independent_group_provenance(request):
    definition = _definition(cells=3, points=True)
    owner = _owner(request, definition)
    old_only = _owner(request, replace(definition, footprints=()))
    np.testing.assert_array_equal(owner._K.toarray(), old_only._K.toarray())
    np.testing.assert_array_equal(owner._C[:6].toarray(), old_only._C.toarray())
    assert owner._footprint_incidence[0].row_start == 6
    result = owner.solve_linear(_load(definition), selected_footprints=
        api.B2NodalRigidFootprintSelection(("right-patch",)))
    assert result.status == "completed", result.failure_reason
    assert tuple(n for n, _ in result.point_multipliers) == ("old-point",)
    assert tuple(w.footprint_id for w in result.footprint_wrenches) == ("right-patch",)


@pytest.mark.parametrize("count", (1, 10, 100))
def test_selected_batches_one_factor_and_no_unselected_recovery(request, monkeypatch, count):
    definition = _definition()
    owner = _owner(request, definition)
    first = owner.solve_linear(_load(definition), recover_members={})
    assert first.status == "completed", first.failure_reason
    selection = api.B2NodalRigidFootprintSelection(("right-patch",))
    cases = tuple((f"case-{i}", _load(definition)*(i+1)) for i in range(count))
    result = owner.solve_linear_many(cases, selected_dofs=(len(first.displacements)-5,),
        recover_members={}, selected_footprints=selection)
    assert type(result) is api.ExperimentalB2Q4FootprintBatchResult
    assert result.status == "completed", result.failure_reason
    assert result.factorization_count == 1 and result.solve_count == count+1
    assert len(result.cases) == count
    for i, case in enumerate(result.cases):
        assert case.case_id == f"case-{i}" and case.beam_resultants_local == ()
        np.testing.assert_allclose(case.selected_displacements,
            (first.displacements[-5]*(i+1),), rtol=1e-8, atol=1e-12)
        assert len(case.footprint_wrenches) == 1
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as panel
    def forbidden(*args):
        raise AssertionError("unselected group recovery materialized")
    monkeypatch.setattr(panel, "selected_wrenches", forbidden)
    assert owner.solve_linear(_load(definition), recover_members={}).status == "completed"
    omitted = owner.solve_linear_many((cases[0],), selected_dofs=(len(first.displacements)-5,), recover_members={})
    assert omitted.status == "completed" and omitted.cases[0].footprint_wrenches == ()


@pytest.mark.parametrize("stop", ("cancel", "close", "failure"))
def test_batch_complete_prefix_and_callback_lifetime(request, stop):
    definition = _definition(); owner = _owner(request, definition)
    token = CancellationToken()
    if stop == "failure":
        original = owner._handle
        class FailSecond:
            @property
            def factorization_count(self): return original.factorization_count
            @property
            def solve_count(self): return original.solve_count
            def solve(self, rhs):
                if original.solve_count == 1:
                    raise RuntimeError("injected B2 footprint backend failure")
                return original.solve(rhs)
        owner._handle = FailSecond()
    submitted = [[name, _load(definition)] for name in ("first", "second", "last")]
    def completed(name):
        assert name == "first"
        if stop == "cancel": token.cancel()
        if stop == "close": owner.close()
        submitted[1][0] = "caller-mutated"
    result = owner.solve_linear_many(submitted, selected_dofs=(49,), recover_members={},
        selected_footprints=api.B2NodalRigidFootprintSelection(("right-patch",)),
        cancellation_token=token, on_case=completed if stop != "failure" else None)
    assert result.status == ("failed" if stop == "failure" else "cancelled")
    assert tuple(c.case_id for c in result.cases) == ("first",)
    assert result.factorization_count == 1 and result.solve_count == 1
    assert len(result.cases[0].footprint_wrenches) == 1
    if stop == "close":
        assert owner._handle is None
        owner.close()


def test_caller_ownership_and_selected_only_membership(request):
    definition = _definition(selected_only=True)
    nodes = {n: list(xyz) for n, xyz in definition.shell_nodes_m.items()}
    membership = list(definition.footprints[0].shell_nodes)
    groups = [api.B2NodalRigidFootprint("owned", membership, 11)]
    supplied = replace(definition, shell_nodes_m=nodes, footprints=groups)
    owner = _owner(request, supplied)
    K = owner._K.copy(); C = owner._C.copy(); T = owner._T.copy()
    membership.clear(); groups.clear(); nodes[2][0] = 999.
    supplied.transformation.data[:] = 999.
    # The owned prescribed-offset snapshot is deliberately read-only: caller
    # mutation is refused by the view itself, which is the isolation proof.
    original_offset = np.array(supplied.prescribed_offset)
    assert supplied.prescribed_offset.flags.writeable is False
    with pytest.raises(ValueError):
        supplied.prescribed_offset[:] = 999.
    with pytest.raises(ValueError):
        supplied.prescribed_offset[0] = 999.
    np.testing.assert_array_equal(supplied.prescribed_offset, original_offset)
    np.testing.assert_array_equal(owner._K.data, K.data)
    np.testing.assert_array_equal(owner._C.data, C.data)
    np.testing.assert_array_equal(owner._T.data, T.data)
    assert owner.footprints[0].shell_nodes == definition.footprints[0].shell_nodes
    with pytest.raises(FrozenInstanceError): owner.footprints[0].beam_node = 99
    # The fourth cell vertex has no group row; no continuous-area claim.
    assert all(i.shell_node != definition.shells[-1].node_ids[-1] for i in owner._footprint_incidence)


@pytest.mark.parametrize("bad", ("unknown", "degenerate", "duplicate", "overlap", "incomplete", "wrong_type", "supported", "point_overlap", "id_collision"))
def test_explicit_admission_negatives(bad):
    definition = _definition()
    group = definition.footprints[0]
    if bad == "unknown": definition = replace(definition, footprints=(replace(group, beam_node=999),))
    elif bad == "degenerate": definition = replace(definition, footprints=(replace(group, shell_nodes=(1, 2, 3), patch_cells=()),))
    elif bad == "duplicate": definition = replace(definition, footprints=(group, group))
    elif bad == "overlap": definition = replace(definition, footprints=(group, replace(group, footprint_id="overlap")))
    elif bad == "incomplete": definition = replace(definition, footprints=(replace(group, shell_nodes=group.shell_nodes[:3]),))
    elif bad == "wrong_type": definition = replace(definition, footprints=(api.NodalRigidFootprint("wrong", group.shell_nodes, 11),))
    elif bad == "supported":
        # Three fixed selected poses leave 18 rows acting only on one free
        # six-coordinate beam pose. Area admission does not prove rank.
        fixed_nodes = group.shell_nodes[:3]
        selected = set(6*(n-1)+j for n in fixed_nodes for j in range(6))
        free = [i for i in range(54) if i not in selected]
        definition = replace(definition, transformation=sparse.eye(54, format="csr")[:, free],
            footprints=(replace(group, shell_nodes=fixed_nodes, patch_cells=()),))
    elif bad == "point_overlap":
        definition = replace(definition, points=(api.Q4B2Point("old", "shell-1", "beam-1", .5),))
    else:
        definition = _definition(cells=3, points=True)
        definition = replace(definition, footprints=(replace(definition.footprints[0], footprint_id="old-point"),))
    with pytest.raises(ValueError): api.create_analysis(definition=definition)


def test_static_capability_and_strict_selected_types(request):
    owner = _owner(request, _definition())
    assert owner.capabilities.footprint_law_id == LAW_ID
    assert owner.capabilities.production_qualified is False
    assert "modes" not in owner.capabilities.operations
    with pytest.raises(ValueError, match="explicit shell density and beam physical mass"):
        owner.modes(api.B2ReferenceModesRequest())
    with pytest.raises(ValueError):
        owner.solve_linear(_load(_definition()), selected_footprints=api.NodalRigidFootprintSelection(("right-patch",)))
    with pytest.raises(ValueError):
        owner.solve_linear(_load(_definition()), selected_footprints=api.B2NodalRigidFootprintSelection(("absent",)))


def test_empty_group_preserves_legacy_result_and_capability_schema(request):
    # Root also runs the captured old-wheel oracle; this checks dispatch and
    # exact default-vs-explicit-empty behavior within the new source.
    definition = _definition(cells=3, points=True)
    empty = replace(definition, footprints=())
    one = _owner(request, empty)
    two = api.ExperimentalB2Q4PanelAnalysis(empty.shell_nodes_m, empty.shells,
        empty.members, empty.points, transformation=empty.transformation,
        prescribed_offset=empty.prescribed_offset)
    request.addfinalizer(two.close)
    np.testing.assert_array_equal(one._K.toarray(), two._K.toarray())
    np.testing.assert_array_equal(one._C.toarray(), two._C.toarray())
    result = one.solve_linear(_load(empty))
    assert result.status == "completed", result.failure_reason
    assert type(result).__name__ == "ExperimentalB2Q4PanelResult"
    assert "footprint_wrenches" not in asdict(result)
    assert "footprints" not in asdict(one.capabilities)
    assert asdict(result) == asdict(two.solve_linear(_load(empty)))


def test_group_moment_retains_sub_ulp_arm_and_rejects_nonfinite():
    # Difference 1 - 2^-54 has a low part. Preserve its cross contribution
    # through the final fsum rather than rounding the arm first.
    xs = (1., 0., 0.); xb = (2.**-54, 0., 0.)
    matrix = reference_shell_matrix(xs, xb)
    item = _Incidence("precision", 1, 10, 0, tuple(range(6)), tuple(range(6, 12)),
                      xs, xb, tuple(tuple(row) for row in matrix))
    lam = np.array((0., 2.**54, 0., 0., 0., 0.))
    group = selected_wrenches((item,), ("precision",), lam, np.zeros(6))[0]
    assert group.shell_moment_Nm[2] == -1.
    with pytest.raises(ValueError):
        selected_wrenches((item,), ("precision",), np.full(6, np.nan), np.zeros(6))


def test_complete_patch_edges_use_actual_cycles_and_shared_endpoint_groups():
    # Cells share two diagonal vertices, not an actual cyclic edge. A caller
    # permutation must not invent connectivity through those two vertices.
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (1., 1., 0.),
             4: (0., 1., 0.), 5: (2., 0., 0.), 6: (0., 2., 0.)}
    shells = (api.Q4PanelShell("one", (1, 2, 3, 4), 1000., .3, .15),
              api.Q4PanelShell("two", (1, 5, 3, 6), 1000., .3, .15))
    si = {n: i for i, n in enumerate(nodes)}; bi = {10: len(si)}
    group = api.B2NodalRigidFootprint("diagonal", tuple(nodes), 10,
        ((1, 3, 2, 4), (1, 3, 5, 6)))
    with pytest.raises(ValueError, match="edge-connected"):
        compile_footprints((group,), nodes, shells, {10: (.5, .5, .2)}, si, bi, ())
    # Repeated beam endpoint is valid for disjoint selected-node groups. The
    # owner subsequently checks the complete supported numerical row rank.
    groups = (api.B2NodalRigidFootprint("a", (1, 2, 3), 10),
              api.B2NodalRigidFootprint("b", (4, 5, 6), 10))
    owned, incidences = compile_footprints(groups, nodes, shells,
        {10: (.5, .5, .2)}, si, bi, ())
    assert tuple(g.footprint_id for g in owned) == ("a", "b")
    assert tuple(i.row_start for i in incidences) == tuple(range(0, 36, 6))
