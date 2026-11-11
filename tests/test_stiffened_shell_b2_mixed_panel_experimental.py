"""Static assembly checks against native local operators, not physics qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as api
from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point, LAW_ID, S3_LAW_ID
from anyfesolver.control import CancellationToken
from anyfesolver.elements import create_shell_element
from anyfesolver.e4_pl_element import FORMULATION_ID as Q4_ID
from anyfesolver.e4_pl_s3_v2d_element import (
    FORMULATION_ID as S3_ID, NativeParityCapabilityError,
)
from anyfesolver.fe_core import FEModel
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as owner_module


def _definition(kind="mixed"):
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    def triangle(name, ids):
        return api.S3V2DPanelShell(name, ids, 1000., .3, .15, (0., 0., 1.))
    left = ((api.Q4PanelShell("left", (1, 2, 5, 4), 1000., .3, .15),)
            if kind != "s3" else
            (triangle("left", (1, 2, 5)), triangle("left-upper", (1, 5, 4))))
    right = ((api.Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15),)
             if kind == "q4" else
             (triangle("right", (2, 3, 6)), triangle("right-upper", (2, 6, 5))))
    section = api.LinearSection.isotropic(
        E=2000., nu=.25, area=.04, Iy=.00015, Iz=.0003,
        J=.0004, ky=.8, kz=.7)
    members = tuple(api.PhysicalB2Member(
        name, name+"-source", (10+i, 11+i),
        ((.2+i, .5, .2), (1.2+i, .5, .2)), (0., 1., 0.), "section", section)
        for i, name in enumerate(("beam-left", "beam-right")))
    points = (api.Q4B2Point("joint-left", "left", "beam-left", .5, (0., .08, -.03)),
              api.Q4B2Point("joint-right", "right", "beam-right", .5, (0., .08, -.03)))
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    fixed.update(6*i+5 for i in (1, 2, 4, 5))
    free = np.array([i for i in range(54) if i not in fixed])
    definition = api.B2MixedPanelDefinition(
        api.B2_E4_PL_FAST_LINEAR_V1, nodes, left+right, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54))
    return definition, free


def _dense_operators(definition):
    """Independent dense scatter of the admitted local operators and point laws."""
    model = FEModel("mixed-panel-assembly-oracle")
    shell_index = {node: i for i, node in enumerate(sorted(definition.shell_nodes_m))}
    beam_ids = sorted({node for member in definition.members for node in member.node_ids})
    beam_index = {node: i+len(shell_index) for i, node in enumerate(beam_ids)}
    for node, xyz in definition.shell_nodes_m.items():
        model.add_node(node, *xyz)
    pieces = []
    for eid, shell in enumerate(definition.shells, 1):
        material = "oracle-"+str(eid)
        model.add_material(material, shell.E, shell.nu)
        formulation = "e4-pl-s3-v2d" if isinstance(shell, api.S3V2DPanelShell) else "e4-pl"
        element = create_shell_element(
            eid, list(shell.node_ids), material, formulation=formulation,
            thickness=shell.thickness_m, reference_normal=shell.reference_normal)
        model.add_element(eid, element)
        pieces.append((shell, element, material))
    K = np.zeros((54, 54))
    C = np.zeros((6*len(definition.points), 54))
    for shell, element, material in pieces:
        slots = [6*shell_index[node]+j for node in shell.node_ids for j in range(6)]
        K[np.ix_(slots, slots)] += element.compute_stiffness_matrix(
            model.mesh, model.get_material(material))
    for member in definition.members:
        frame = member.local_dof_map()
        local = frame.T@member.local_operator().evaluate(np.zeros(12)).tangent@frame
        slots = [6*beam_index[node]+j for node in member.node_ids for j in range(6)]
        K[np.ix_(slots, slots)] += local
    shells = {shell.shell_id: shell for shell in definition.shells}
    members = {member.member_id: member for member in definition.members}
    for index, point in enumerate(definition.points):
        shell, member = shells[point.shell_id], members[point.member_id]
        vertices = np.array([definition.shell_nodes_m[node] for node in shell.node_ids])
        law = compile_reference_point(member, vertices, station=point.station,
                                      offset_local_m=point.offset_local_m)
        slots = ([6*shell_index[node]+j for node in shell.node_ids for j in range(6)] +
                 [6*beam_index[node]+j for node in member.node_ids for j in range(6)])
        C[6*index:6*index+6, slots] = law.constraint
    return K, C


def _load():
    load = np.zeros(54)
    load[-6:] = (.04, .08, -.03, .01, -.02, .015)
    return load


def _oracle(K, C, free, load):
    A = np.block([[K[np.ix_(free, free)], C[:, free].T],
                  [C[:, free], np.zeros((len(C), len(C)))]])
    solution = np.linalg.solve(A, np.r_[load[free], np.zeros(len(C))])
    u = np.zeros(54)
    u[free] = solution[:len(free)]
    lam = solution[len(free):]
    return u, lam, K@u+C.T@lam-load


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_native_local_operator_kkt_work_reactions_and_gaps(kind):
    definition, free = _definition(kind)
    K, C = _dense_operators(definition)
    load = _load()
    expected_u, expected_lam, expected_r = _oracle(K, C, free, load)
    owner = api.create_analysis(definition=definition)
    try:
        result = owner.solve_linear(load)
        assert result.status == "completed", result.failure_reason
        u = np.array(result.displacements)
        lam = np.concatenate([value for _, value in result.point_multipliers])
        np.testing.assert_allclose(owner._K.toarray(), K, rtol=8*np.finfo(float).eps,
                                   atol=8*np.finfo(float).eps*np.max(abs(K)))
        np.testing.assert_allclose(owner._C.toarray(), C, rtol=0., atol=0.)
        np.testing.assert_allclose(u, expected_u, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(lam, expected_lam, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.affine_reactions, expected_r, rtol=1e-8, atol=1e-12)
        assert np.max(abs(C@u)) <= 1e-9
        np.testing.assert_allclose([g for _, g in result.point_gaps], (C@u).reshape(-1, 6),
                                   rtol=0., atol=1e-15)
        assert abs(u@load-2*result.internal_strain_energy) <= 1e-8*abs(u@load)
        np.testing.assert_allclose(result.internal_strain_energy, .5*u@K@u, rtol=1e-8)
        assert result.factorization_count == 1 and result.solve_count == 1
        caps = owner.capabilities
        assert isinstance(caps, api.ExperimentalB2MixedPanelCapabilities)
        assert caps.operations == ("solve_linear", "solve_linear_many", "close")
        assert caps.shell_formulations == tuple(
            (shell.shell_id, S3_ID if isinstance(shell, api.S3V2DPanelShell) else Q4_ID)
            for shell in definition.shells)
        assert caps.point_laws == (("joint-left", S3_LAW_ID if kind == "s3" else LAW_ID),
                                   ("joint-right", S3_LAW_ID))
        assert caps.point_law_id == ((S3_LAW_ID,) if kind == "s3" else (LAW_ID, S3_LAW_ID))
        assert owner._M is None and "deferred" in caps.spectral_domain
        with pytest.raises(ValueError, match="S3 V2D.*deferred"):
            owner.modes(api.B2ReferenceModesRequest())
    finally:
        owner.close()


@pytest.mark.parametrize("count", (1, 10, 100))
@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_selected_batch_and_cancellation_reuse_one_factor(kind, count, monkeypatch):
    definition, free = _definition(kind)
    K, C = _dense_operators(definition)
    calls = []
    original = owner_module.factorize_cached
    def factor(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner_module, "factorize_cached", factor)
    owner = api.create_analysis(definition=definition)
    selected = (49, 50, 52)
    loads = tuple((str(i), (1.+i/count)*_load()) for i in range(count))
    token = CancellationToken()
    try:
        result = owner.solve_linear_many(loads, selected_dofs=selected,
                                          selected_points=("joint-right",))
        assert result.status == "completed", result.failure_reason
        assert len(result.cases) == count
        assert result.factorization_count == 1 and result.solve_count == count
        for case, (_, load) in zip(result.cases, loads):
            u, lam, reactions = _oracle(K, C, free, load)
            np.testing.assert_allclose(case.selected_displacements, u[list(selected)], rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(case.selected_affine_reactions, reactions[list(selected)], rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(case.point_multipliers[0][1], lam[6:], rtol=1e-8, atol=1e-12)
            assert tuple(name for name, _ in case.point_gaps) == ("joint-right",)
        before = owner._handle.solve_count
        def stop(_case_id):
            token.cancel("complete first mixed-shell case")
        cancelled = owner.solve_linear_many(
            (("first", _load()), ("never", _load())), selected_dofs=selected,
            cancellation_token=token, on_case=stop)
        assert cancelled.status == "cancelled" and len(cancelled.cases) == 1
        assert owner._handle.solve_count == before+1
        assert calls == [1]
    finally:
        owner.close()
    assert owner._handle is None


def test_mixed_owner_snapshots_inputs_and_rejects_changed_submitted_load():
    definition, free = _definition()
    caller_nodes = {n: np.array(x) for n, x in definition.shell_nodes_m.items()}
    caller_transform = definition.transformation
    caller_offset = np.array(definition.prescribed_offset)
    definition = replace(definition, shell_nodes_m=caller_nodes,
                         transformation=caller_transform, prescribed_offset=caller_offset)
    K, C = _dense_operators(definition)
    load = _load()
    u, _, reactions = _oracle(K, C, free, load)
    owner = api.create_analysis(definition=definition)
    try:
        caller_nodes[2][:] = (100., 200., 300.)
        caller_transform.data[:] = 7.
        caller_offset[:] = 8.
        definition.transformation.data[:] = 7.
        with pytest.raises(TypeError):
            definition.shell_nodes_m[2] = (100., 200., 300.)
        with pytest.raises(ValueError):
            definition.prescribed_offset[:] = 8.
        result = owner.solve_linear(load)
        assert result.status == "completed", result.failure_reason
        np.testing.assert_allclose(result.displacements, u, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.affine_reactions, reactions, rtol=1e-8, atol=1e-12)
        names = []
        def mutate(name):
            names.append(name)
            load.fill(999.)
        with pytest.raises(ValueError, match="load case changed after submission"):
            owner.solve_linear_many((("a", load), ("b", load)),
                                    selected_dofs=(49, 50), on_case=mutate)
        assert names == ["a"] and owner._handle.solve_count == 2
    finally:
        owner.close()


def test_historical_q4_route_remains_explicit_and_unchanged():
    definition, _ = _definition("q4")
    legacy = api.B2Q4PanelDefinition(**definition.__dict__)
    owners = [api.create_analysis(definition=item) for item in (legacy, definition)]
    try:
        old, mixed = owners
        assert type(old) is owner_module.ExperimentalB2Q4PanelAnalysis
        assert type(old.capabilities) is owner_module.ExperimentalB2Q4PanelCapabilities
        assert old.capabilities.point_law_id == LAW_ID
        assert old.solve_linear(_load()) == mixed.solve_linear(_load())
    finally:
        for owner in owners:
            owner.close()
    s3, _ = _definition("s3")
    with pytest.raises(ValueError, match="multi-Q4/multi-B2"):
        api.create_analysis(definition=api.B2Q4PanelDefinition(**s3.__dict__))


@pytest.mark.parametrize("normal", ((0., 0., 0.), (1., 0., 0.), (0., 0., float("nan"))))
def test_native_s3_reference_normal_guard_is_not_bypassed(normal):
    definition, _ = _definition("s3")
    shells = (replace(definition.shells[0], reference_normal=normal),)+definition.shells[1:]
    with pytest.raises((ValueError, NativeParityCapabilityError)):
        api.create_analysis(definition=replace(definition, shells=shells))


@pytest.mark.parametrize("nodes", ((1, 1, 5), (1, 2, 3), (1, 2, 999), (1, 2, 5, 4),
                                  (np.int64(1), 2, 5)))
def test_native_s3_connectivity_and_degeneracy_guards(nodes):
    definition, _ = _definition("s3")
    shells = (replace(definition.shells[0], node_ids=nodes),)+definition.shells[1:]
    with pytest.raises((ValueError, NativeParityCapabilityError)):
        api.create_analysis(definition=replace(definition, shells=shells))

def test_native_s3_node_sequence_input_is_normalized_to_owned_tuple():
    # The ownership layer snapshots caller sequences into owned tuples;
    # a valid list is admitted normalized, not refused as malformed.
    definition, _ = _definition("s3")
    shells = (replace(definition.shells[0], node_ids=[1, 2, 5]),)+definition.shells[1:]
    owned = replace(definition, shells=shells)
    assert owned.shells[0].node_ids == (1, 2, 5)
    assert type(owned.shells[0].node_ids) is tuple
    owner = api.create_analysis(definition=owned)
    owner.close()


def test_s3_nodal_footprints_fail_before_factorization(monkeypatch):
    definition, _ = _definition()
    footprint = api.B2NodalRigidFootprint("unsupported", (2, 3, 6), 12)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported footprint reached factorization")
    monkeypatch.setattr(owner_module, "factorize_cached", forbidden)
    with pytest.raises(ValueError, match="S3 mixed-panel nodal footprints are unsupported"):
        api.create_analysis(definition=replace(definition, footprints=(footprint,)))


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_mixed_route_retains_failed_solve_validation(kind, monkeypatch):
    definition, _ = _definition(kind)
    owner = api.create_analysis(definition=definition)
    try:
        monkeypatch.setattr(owner._handle, "solve", lambda rhs: np.zeros_like(rhs))
        result = owner.solve_linear(_load())
        assert result.status == "failed" and result.failure_reason
        assert result.displacements == () and not result.production_qualified
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_native_s3_normal_scaling_is_normalized_without_changing_response(kind):
    definition, _ = _definition(kind)
    scaled = replace(definition, shells=tuple(
        replace(shell, reference_normal=(0., 0., 2.))
        if isinstance(shell, api.S3V2DPanelShell) else shell
        for shell in definition.shells))
    owners = [api.create_analysis(definition=item) for item in (definition, scaled)]
    try:
        original, normalized = owners
        np.testing.assert_array_equal(original._K.toarray(), normalized._K.toarray())
        assert original.solve_linear(_load()) == normalized.solve_linear(_load())
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_reversed_s3_connectivity_keeps_authoritative_normal_and_physical_response(kind):
    definition, free = _definition(kind)
    reversed_definition = replace(definition, shells=tuple(
        replace(shell, node_ids=(shell.node_ids[0], shell.node_ids[2], shell.node_ids[1]))
        if isinstance(shell, api.S3V2DPanelShell) else shell
        for shell in definition.shells))
    original_K, _ = _dense_operators(definition)
    reversed_K, reversed_C = _dense_operators(reversed_definition)
    scale = 8*np.finfo(float).eps*np.max(abs(original_K))
    np.testing.assert_allclose(reversed_K, original_K, rtol=8*np.finfo(float).eps, atol=scale)
    owners = [api.create_analysis(definition=item) for item in (definition, reversed_definition)]
    try:
        original, reversed_owner = owners
        result = reversed_owner.solve_linear(_load())
        baseline = original.solve_linear(_load())
        assert result.status == baseline.status == "completed", (baseline.failure_reason, result.failure_reason)
        np.testing.assert_allclose(reversed_owner._K.toarray(), reversed_K,
                                   rtol=8*np.finfo(float).eps, atol=scale)
        np.testing.assert_allclose(reversed_owner._C.toarray(), reversed_C, rtol=0., atol=0.)
        u, _, reactions = _oracle(reversed_K, reversed_C, free, _load())
        np.testing.assert_allclose(result.displacements, u, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.affine_reactions, reactions, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.displacements, baseline.displacements, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.affine_reactions, baseline.affine_reactions, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(result.internal_strain_energy, baseline.internal_strain_energy, rtol=1e-8)
        assert np.max(abs(reversed_C@np.array(result.displacements))) <= 1e-9
        # Connectivity changes the shell chart and multiplier components; they
        # are work-conjugate frame components, not an invariant global wrench.
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_nonzero_affine_offset_dense_kkt_reactions_and_support_work(kind):
    definition, free = _definition(kind)
    change = sparse.eye(len(free), format="lil")
    change[0, 1] = .17
    transformation = (definition.transformation@change.tocsr()).tocsr()
    offset = np.zeros(54)
    offset[:6] = np.array((2., -1., 3., 1., -2., .5))*1e-5
    offset[18:24] = np.array((0., .5, -1., .3, .2, -.4))*1e-5
    offset[36:42] = np.array((-1., .5, 2., -.2, .1, .3))*1e-5
    definition = replace(definition, transformation=transformation, prescribed_offset=offset)
    K, C = _dense_operators(definition)
    T = transformation.toarray()
    load = _load()
    reduced_C = C@T
    A = np.block([[T.T@K@T, reduced_C.T],
                  [reduced_C, np.zeros((len(C), len(C)))]])
    rhs = np.r_[T.T@(load-K@offset), -C@offset]
    assert np.any(C@offset != 0.) and np.any(K@offset != 0.)
    answer = np.linalg.solve(A, rhs)
    expected_u = T@answer[:T.shape[1]]+offset
    expected_lam = answer[T.shape[1]:]
    expected_reactions = K@expected_u+C.T@expected_lam-load
    owner = api.create_analysis(definition=definition)
    try:
        result = owner.solve_linear(load)
        assert result.status == "completed", result.failure_reason
        u = np.array(result.displacements)
        reactions = np.array(result.affine_reactions)
        lam = np.concatenate([value for _, value in result.point_multipliers])
        np.testing.assert_allclose(u, expected_u, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(lam, expected_lam, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(reactions, expected_reactions, rtol=1e-8, atol=1e-12)
        assert np.max(abs(C@u)) <= 1e-9
        assert np.max(abs(T.T@reactions)) <= 1e-8*np.max(abs(load)+abs(K@u)+abs(C.T@lam))
        work = u@load+offset@reactions
        assert abs(2*result.internal_strain_energy-work) <= 1e-8*abs(work)
        np.testing.assert_allclose(result.internal_strain_energy, .5*u@K@u, rtol=1e-8)
        assert result.factorization_count == result.solve_count == 1
    finally:
        owner.close()
