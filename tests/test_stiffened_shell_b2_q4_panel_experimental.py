"""Two Q4s, two native physical B2 members, and two F10 point joints."""

import numpy as np
import pytest
import weakref
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from scipy import sparse
from scipy.linalg import null_space
from anyfesolver import stiffened_shell

from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2Q4PanelAnalysis, ExperimentalB2Q4PanelError,
    Q4B2Point, Q4PanelShell,
    _NumericalFailure, _require_independent_supported_point_rows,
    _scaled_point_kkt,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.linalg import MatrixClass, factorize_cached
from test_b2_f09_physical_off_axis_elbow import (
    _high_precision_rounded_matrix_solve, _physical_section,
)


def _panel():
    nodes = {
        1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
        4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.),
    }
    shells = (
        Q4PanelShell("left", (1, 2, 5, 4), 1000., .3, .15),
        Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15),
    )
    section = LinearSection.isotropic(
        E=2000., nu=.25, area=.04, Iy=.00015, Iz=.0003,
        J=.0004, ky=.8, kz=.7)
    members = (
        PhysicalB2Member("beam-left", "beam-source-left", (10, 11),
                         ((.2, .5, .2), (1.2, .5, .2)),
                         (0., 1., 0.), "beam-section", section),
        PhysicalB2Member("beam-right", "beam-source-right", (11, 12),
                         ((1.2, .5, .2), (2.2, .5, .2)),
                         (0., 1., 0.), "beam-section", section),
    )
    points = (
        Q4B2Point("joint-left", "left", "beam-left", .5,
                  (0., .08, -.03)),
        Q4B2Point("joint-right", "right", "beam-right", .5,
                  (0., .08, -.03)),
    )
    shell_ids = tuple(sorted(nodes))
    beam_ids = (10, 11, 12)
    shell_index = {node_id: i for i, node_id in enumerate(shell_ids)}
    beam_index = {node_id: i+len(shell_ids)
                  for i, node_id in enumerate(beam_ids)}
    fixed = set()
    for node_id in (1, 4):
        fixed.update(6*shell_index[node_id]+j for j in range(6))
    fixed.update(6*beam_index[10]+j for j in range(6))
    for node_id in (2, 3, 5, 6):
        fixed.add(6*shell_index[node_id]+5)
    free = np.array([i for i in range(54) if i not in fixed])
    return nodes, shells, members, points, free


def _independent_dense(nodes, shells, members, points):
    """Assemble from local operators without calling the panel owner."""
    model = FEModel("independent-panel")
    for node_id, position in nodes.items():
        model.add_node(node_id, *position)
    shell_index = {node_id: i for i, node_id in enumerate(sorted(nodes))}
    beam_index = {node_id: i+len(nodes)
                  for i, node_id in enumerate((10, 11, 12))}
    stiffness = np.zeros((54, 54))
    constraint = np.zeros((12, 54))
    for index, shell in enumerate(shells, 1):
        material = f"oracle-{index}"
        model.add_material(material, shell.E, shell.nu)
        element = QualifiedE4PLShellElement(
            index, shell.node_ids, material, thickness=shell.thickness_m,
            reference_normal=np.array(shell.reference_normal))
        model.add_element(index, element)
        local = element.compute_stiffness_matrix(
            model.mesh, model.get_material(material))
        slots = np.array([6*shell_index[node_id]+j
                          for node_id in shell.node_ids for j in range(6)])
        stiffness[np.ix_(slots, slots)] += local
    for member in members:
        frame = member.local_dof_map()
        local = frame.T @ member.local_operator().evaluate(
            np.zeros(12)).tangent @ frame
        slots = np.array([6*beam_index[node_id]+j
                          for node_id in member.node_ids for j in range(6)])
        stiffness[np.ix_(slots, slots)] += local
    shell_by_id = {shell.shell_id: shell for shell in shells}
    member_by_id = {member.member_id: member for member in members}
    for index, point in enumerate(points):
        shell = shell_by_id[point.shell_id]
        member = member_by_id[point.member_id]
        vertices = np.array([nodes[node_id] for node_id in shell.node_ids])
        law = compile_reference_point(
            member, vertices, station=point.station,
            offset_local_m=point.offset_local_m)
        slots = np.array(
            [6*shell_index[node_id]+j for node_id in shell.node_ids
             for j in range(6)] +
            [6*beam_index[node_id]+j for node_id in member.node_ids
             for j in range(6)])
        constraint[6*index:6*index+6, slots] += law.constraint
    return stiffness, constraint


def test_supported_point_row_audit_bounds_dense_work(monkeypatch):
    small = sparse.csr_matrix(((1., 2.), (0, 0), (0, 2)), shape=(1, 3))
    _require_independent_supported_point_rows(small)
    with pytest.raises(ValueError, match="independent supported point rows"):
        _require_independent_supported_point_rows(
            sparse.csr_matrix(((1., 2.), ([0, 1], [0, 0])), shape=(2, 2)))

    def forbid_dense(_self, *args, **kwargs):
        raise AssertionError("large supported-row audit materialized a dense matrix")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbid_dense)
    large = sparse.eye(96, 192, format="csc")
    _require_independent_supported_point_rows(large)
    dependent = large.tolil()
    dependent[1, :] = 0.
    dependent[1, 0] = 2.
    with pytest.raises(ValueError, match="structurally independent"):
        _require_independent_supported_point_rows(dependent.tocsc())

    # Structural rank alone cannot detect proportional rows with two distinct
    # columns; the production sparse KKT factorization must reject them.
    proportional = sparse.eye(60, 61, format="lil")
    proportional[0, 1] = 1.
    proportional[1, :] = 0.
    proportional[1, 0] = 2.
    proportional[1, 1] = 2.
    proportional = proportional.tocsc()
    _require_independent_supported_point_rows(proportional)
    kkt = sparse.bmat(((sparse.eye(61, format="csc"), proportional.T),
                       (proportional, None)), format="csc")
    factor = factorize_cached(kkt, MatrixClass.SYMMETRIC_INDEFINITE)
    assert factor.status == "failed"


def test_kkt_scaling_retains_constraint_anchored_zero_energy_coordinate():
    stiffness = sparse.diags((4., 0.), format="csc")
    constraint = sparse.csr_matrix(((1e-9,), ((0,), (1,))), shape=(1, 2))
    scaled, factors = _scaled_point_kkt(stiffness, constraint)
    physical_rhs = np.array((8., 3e-6, 2e-9))
    solution = factors*np.linalg.solve(
        scaled.toarray(), factors*physical_rhs)
    np.testing.assert_allclose(solution, (2., 2., 3000.), rtol=1e-14)
    original = sparse.bmat(((stiffness, constraint.T),
                            (constraint, None)), format="csr")
    np.testing.assert_allclose(original@solution, physical_rhs, rtol=1e-14)
    with pytest.raises(ExperimentalB2Q4PanelError,
                       match="unanchored zero-energy"):
        _scaled_point_kkt(stiffness, sparse.csr_matrix((1, 2)))


@pytest.mark.parametrize("offset_sign", (1., -1.))
def test_multi_q4_multi_b2_panel_matches_bounded_dense_oracle(offset_sign):
    nodes, shells, members, points, free = _panel()
    points = tuple(replace(point, offset_local_m=tuple(
        offset_sign*value for value in point.offset_local_m)) for point in points)
    T = sparse.eye(54, format="csr")[:, free]
    u0 = np.zeros(54)
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points, T, u0))
    assert analysis.shell_node_ids == (1, 2, 3, 4, 5, 6)
    assert analysis.beam_node_ids == (10, 11, 12)
    assert analysis.capabilities.status == "experimental"
    T.data[:] = 0.
    u0[:] = 1.
    stiffness, constraint = _independent_dense(nodes, shells, members, points)
    c = constraint[:, free]
    z = null_space(c)
    reduced_k = stiffness[np.ix_(free, free)]
    for scale in (1., 1.7):
        load = np.zeros(54)
        load[6*2+2] = -.12*scale
        load[6*5+2] = -.16*scale
        load[6*8+1] = .08*scale
        result = analysis.solve_linear(load)
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == 1
        u = np.asarray(result.displacements)
        lam = np.concatenate([row for _, row in result.point_multipliers])
        expected = z @ np.linalg.solve(z.T @ reduced_k @ z,
                                        z.T @ load[free])
        expected_lam = np.linalg.lstsq(
            c.T, load[free]-reduced_k @ expected, rcond=None)[0]
        np.testing.assert_allclose(u[free], expected, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(lam, expected_lam, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(constraint @ u, 0., atol=1e-10)
        reaction = np.asarray(result.affine_reactions)
        np.testing.assert_allclose(reaction[free], 0., atol=1e-9)
        assert np.linalg.norm(reaction) > 0.
        assert abs(u @ load - 2*result.internal_strain_energy) < 1e-9
        assert len(result.beam_resultants_local) == 2
        assert len(result.point_gaps) == 2
    analysis.close()


@pytest.mark.parametrize("changed_dependency", (
    "section", "geometry", "point", "support",
))
def test_panel_owns_shell_coordinates_and_rebuilds_changed_dependencies(
        changed_dependency):
    nodes, shells, members, points, free = _panel()
    caller_nodes = {node_id: np.asarray(point, dtype=float).copy()
                    for node_id, point in nodes.items()}
    original_nodes = {node_id: point.copy()
                      for node_id, point in caller_nodes.items()}
    transformation = sparse.eye(54, format="csr")[:, free]
    owner = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            caller_nodes, shells, members, points, transformation, np.zeros(54)))
    rebuilt = None
    try:
        load = np.zeros(54)
        load[6*8+1] = .08
        load[6*5+2] = -.16
        first = owner.solve_linear(load)
        assert first.status == "completed"
        original_k = owner._K.copy()
        original_c = owner._C.copy()
        original_t = owner._T.copy()
        # Caller arrays remain mutable; the compiled shell geometry must not.
        for point in caller_nodes.values():
            point[0] += 100.
        unchanged = owner.solve_linear(load)
        for field in ("displacements", "point_multipliers", "affine_reactions"):
            assert getattr(unchanged, field) == getattr(first, field)

        if changed_dependency == "section":
            section = LinearSection.isotropic(
                E=2500., nu=.25, area=.04, Iy=.00015, Iz=.0003,
                J=.0004, ky=.8, kz=.7)
            members = tuple(replace(member, section=section) for member in members)
        elif changed_dependency == "geometry":
            # Widen only the right planar Q4; beam topology and frames stay fixed.
            original_nodes[3][0] += .15
            original_nodes[6][0] += .15
        elif changed_dependency == "point":
            points = (points[0], replace(points[1], station=.7))
        else:
            # A physical MPC replaces the free beam-tip Fy coordinate by
            # one quarter of shell-node 5 Fy; this changes the admissible space.
            retained = np.array([dof for dof in free if dof != 6*8+1])
            transformation = sparse.eye(54, format="lil")[:, retained]
            transformation[6*8+1, int(np.flatnonzero(retained == 6*4+1)[0])] = .25
            transformation = transformation.tocsr()
        rebuilt = stiffened_shell.create_analysis(definition=
            stiffened_shell.B2Q4PanelDefinition(
                stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
                original_nodes, shells, members, points,
                transformation, np.zeros(54)))
        changed = rebuilt.solve_linear(load)
        assert changed.status == "completed", changed.failure_reason
        assert changed.factorization_count == 1
        assert rebuilt._handle is not owner._handle
        stiffness, constraint = _independent_dense(
            original_nodes, shells, members, points)
        basis = transformation.toarray()
        compatible = basis@null_space(constraint@basis)
        expected = compatible@np.linalg.solve(
            compatible.T@stiffness@compatible, compatible.T@load)
        np.testing.assert_allclose(changed.displacements, expected,
                                   rtol=2e-9, atol=2e-10)
        before = np.asarray(first.displacements)
        assert np.linalg.norm(np.asarray(changed.displacements)-before) > (
            1e-8*np.linalg.norm(before))
        # Rebuilding never rewrites the previous owner's compiled dependency set.
        assert (owner._K != original_k).nnz == 0
        assert (owner._C != original_c).nnz == 0
        assert (owner._T != original_t).nnz == 0
        prior = owner.solve_linear(load)
        assert prior.displacements == first.displacements
    finally:
        if rebuilt is not None:
            rebuilt.close()
        owner.close()


def test_panel_batch_owns_submitted_case_identifier_inventory():
    nodes, shells, members, points, free = _panel()
    owner = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    load = np.zeros(54)
    load[6*8+1] = .08
    cases = [["first", load], ["second", load.copy()]]

    def mutate_caller_inventory(case_id):
        if case_id == "first":
            cases[1][0] = "changed-after-submission"
            cases.append(["new-after-submission", load.copy()])

    try:
        result = owner.solve_linear_many(
            cases, selected_dofs=(6*8+1,), on_case=mutate_caller_inventory)
        assert result.status == "completed"
        assert tuple(case.case_id for case in result.cases) == ("first", "second")
        assert result.factorization_count == 1
        assert result.solve_count == 2
    finally:
        owner.close()


@pytest.mark.parametrize("count", (1, 10, 100))
def test_point_panel_batch_reuses_factorization_and_selects_recovery(count):
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    selected = (6*8+1, 6*5+2)
    cases = []
    for index in range(count):
        load = np.zeros(54)
        load[6*8+1] = .08*(1.+index/count)
        load[6*5+2] = -.16*(1.+index/count)
        cases.append((f"load-{index:03d}", load))
    result = analysis.solve_linear_many(
        cases, selected_dofs=selected, selected_points=("joint-right",),
        recover_members={"beam-right": (.25, .75)})
    assert result.status == "completed"
    assert len(result.cases) == count
    assert tuple(case.case_id for case in result.cases) == tuple(
        name for name, _ in cases)
    assert result.factorization_count == 1
    assert result.solve_count == count
    assert result.maximum_rhs_chunk == 1
    assert result.selected_points == ("joint-right",)
    assert all(len(case.selected_displacements) == 2 and
               len(case.selected_affine_reactions) == 2 and
               len(case.beam_resultants_local) == 1 and
               case.beam_resultants_local[0][0] == "beam-right" and
               len(case.beam_resultants_local[0][1]) == 2
               for case in result.cases)
    assert len(result.member_resultant_envelopes) == 12
    assert tuple((item.member_id, item.station_index, item.station_normalized,
                  item.component) for item in result.member_resultant_envelopes) == tuple(
        ("beam-right", station_index, station, component)
        for station_index, station in enumerate((.25, .75))
        for component in ("N", "Qy", "Qz", "Mx", "My", "Mz"))
    single = analysis.solve_linear(
        cases[-1][1], recover_members={"beam-right": (.25, .75)})
    assert single.status == "completed"
    np.testing.assert_allclose(
        result.cases[-1].selected_displacements,
        np.asarray(single.displacements)[list(selected)], rtol=0., atol=0.)
    np.testing.assert_allclose(
        result.cases[-1].selected_affine_reactions,
        np.asarray(single.affine_reactions)[list(selected)], rtol=0., atol=0.)
    assert result.cases[-1].point_multipliers == single.point_multipliers[1:]
    assert result.cases[-1].point_gaps == single.point_gaps[1:]
    assert result.cases[-1].internal_strain_energy == single.internal_strain_energy
    assert analysis._handle.factorization_count == 1
    analysis.close()


def test_point_panel_selected_envelopes_report_signed_extrema_and_first_case_ties():
    nodes, shells, members, points, free = _panel()
    definition = stiffened_shell.B2Q4PanelDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54))
    owner = stiffened_shell.create_analysis(definition=definition)
    control = stiffened_shell.create_analysis(definition=definition)
    selected = (6*8+1, 1, 6*5+2)
    load = np.zeros(54)
    load[6*8+1] = .08
    load[6*5+2] = -.16
    cases = (("z-first", load), ("a-reversed", -2*load),
             ("y-tied", load.copy()), ("zero", np.zeros(54)))
    recovery = {"beam-right": (.25, .75)}
    try:
        expected = tuple(control.solve_linear(value, recover_members=recovery)
                         for _, value in cases)
        assert all(case.status == "completed" for case in expected)
        result = owner.solve_linear_many(cases, selected_dofs=selected,
                                        recover_members=recovery)
        assert result.status == "completed"
        assert tuple(case.case_id for case in result.cases) == tuple(
            case_id for case_id, _ in cases)
        assert result.factorization_count == 1 and result.solve_count == 4
        for envelopes, field in ((result.displacement_envelopes, "displacements"),
                                 (result.affine_reaction_envelopes, "affine_reactions")):
            assert tuple(item.selected_dof for item in envelopes) == selected
            for envelope, dof in zip(envelopes, selected):
                values = tuple(getattr(case, field)[dof] for case in expected)
                minimum_index = min(range(len(values)), key=values.__getitem__)
                maximum_index = max(range(len(values)), key=values.__getitem__)
                assert envelope.minimum == values[minimum_index]
                assert envelope.maximum == values[maximum_index]
                assert envelope.minimum_case_id == cases[minimum_index][0]
                assert envelope.maximum_case_id == cases[maximum_index][0]
        assert len(result.member_resultant_envelopes) == 12
        assert isinstance(result.member_resultant_envelopes[0],
                          stiffened_shell.ExperimentalB2Q4PanelMemberResultantEnvelope)
        for station_index, station in enumerate(recovery["beam-right"]):
            for component_index, component in enumerate(("N", "Qy", "Qz", "Mx", "My", "Mz")):
                envelope = result.member_resultant_envelopes[6*station_index+component_index]
                values = tuple(dict(case.beam_resultants_local)["beam-right"]
                               [station_index][component_index] for case in expected)
                minimum_index = min(range(len(values)), key=values.__getitem__)
                maximum_index = max(range(len(values)), key=values.__getitem__)
                assert (envelope.member_id, envelope.station_index,
                        envelope.station_normalized, envelope.component) == (
                            "beam-right", station_index, station, component)
                assert (envelope.minimum, envelope.minimum_case_id) == (
                    values[minimum_index], cases[minimum_index][0])
                assert (envelope.maximum, envelope.maximum_case_id) == (
                    values[maximum_index], cases[maximum_index][0])
        # The supported uy response ties at zero: preserve submission order,
        # rather than sorting IDs or choosing the last equally governing case.
        ground = result.displacement_envelopes[1]
        assert ground.minimum == ground.maximum == 0.
        assert ground.minimum_case_id == ground.maximum_case_id == "z-first"
        assert abs(expected[0].affine_reactions[1]) > 0.
        with pytest.raises(FrozenInstanceError):
            ground.minimum = 1.
        with pytest.raises(FrozenInstanceError):
            result.member_resultant_envelopes[0].minimum = 1.
    finally:
        owner.close()
        control.close()


@pytest.mark.parametrize("termination", ("cancel", "close", "failure"))
def test_point_panel_selected_envelopes_use_only_complete_prefix(termination, monkeypatch):
    nodes, shells, members, points, free = _panel()
    definition = stiffened_shell.B2Q4PanelDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54))
    owner = stiffened_shell.create_analysis(definition=definition)
    control = stiffened_shell.create_analysis(definition=definition)
    selected = (6*8+1, 1, 6*5+2)
    load = np.zeros(54)
    load[6*8+1] = .08
    load[6*5+2] = -.16
    cases = (("first", load), ("second", -2*load),
             ("unpublished-large", 100*load))
    recovery = {"beam-right": (.25, .75)}
    token = CancellationToken()
    seen = []

    def after_case(case_id):
        seen.append(case_id)
        if len(seen) == 2:
            if termination == "cancel":
                token.cancel("complete envelope prefix")
            elif termination == "close":
                owner.close()

    try:
        expected = tuple(control.solve_linear(value, recover_members=recovery)
                         for _, value in cases[:2])
        assert all(case.status == "completed" for case in expected)
        if termination == "failure":
            original = owner._handle.solve

            def fail_third(rhs):
                if len(seen) == 2:
                    raise RuntimeError("injected envelope backend failure")
                return original(rhs)

            monkeypatch.setattr(owner._handle, "solve", fail_third)
        result = owner.solve_linear_many(
            cases, selected_dofs=selected, recover_members=recovery,
            cancellation_token=token, on_case=after_case)
        assert result.status == ("failed" if termination == "failure" else "cancelled")
        assert tuple(case.case_id for case in result.cases) == ("first", "second")
        assert result.factorization_count == 1 and result.solve_count == 2
        for envelopes, field in ((result.displacement_envelopes, "displacements"),
                                 (result.affine_reaction_envelopes, "affine_reactions")):
            for envelope, dof in zip(envelopes, selected):
                values = tuple(getattr(case, field)[dof] for case in expected)
                minimum_index = min(range(2), key=values.__getitem__)
                maximum_index = max(range(2), key=values.__getitem__)
                assert envelope.minimum == values[minimum_index]
                assert envelope.maximum == values[maximum_index]
                assert envelope.minimum_case_id == cases[minimum_index][0]
                assert envelope.maximum_case_id == cases[maximum_index][0]
        assert len(result.member_resultant_envelopes) == 12
        for envelope in result.member_resultant_envelopes:
            component_index = ("N", "Qy", "Qz", "Mx", "My", "Mz").index(envelope.component)
            values = tuple(dict(case.beam_resultants_local)[envelope.member_id]
                           [envelope.station_index][component_index] for case in expected)
            minimum_index = min(range(2), key=values.__getitem__)
            maximum_index = max(range(2), key=values.__getitem__)
            assert (envelope.minimum, envelope.minimum_case_id) == (
                values[minimum_index], cases[minimum_index][0])
            assert (envelope.maximum, envelope.maximum_case_id) == (
                values[maximum_index], cases[maximum_index][0])
        if termination == "failure":
            assert result.failure_reason == "unpublished-large: injected envelope backend failure"
        elif termination == "close":
            assert owner._handle is None
            assert result.failure_reason == "experimental panel closed after second"
    finally:
        owner.close()
        control.close()


@pytest.mark.parametrize("termination", ("cancel", "failure"))
def test_point_panel_selected_envelopes_are_empty_for_zero_complete_cases(
        termination, monkeypatch):
    nodes, shells, members, points, free = _panel()
    owner = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    load = np.zeros(54)
    load[6*8+1] = .08
    token = CancellationToken()
    if termination == "cancel":
        token.cancel("no envelope cases")
    else:
        def failed_backend(_rhs):
            raise RuntimeError("injected first envelope case failure")

        monkeypatch.setattr(owner._handle, "solve", failed_backend)
    try:
        result = owner.solve_linear_many(
            (("first", load),), selected_dofs=(6*8+1,),
            recover_members={"beam-right": (.25,)}, cancellation_token=token)
        assert result.status == ("cancelled" if termination == "cancel" else "failed")
        assert result.cases == ()
        assert (result.displacement_envelopes == result.affine_reaction_envelopes ==
                result.member_resultant_envelopes == ())
        assert result.solve_count == 0
    finally:
        owner.close()


def test_point_panel_batch_cancellation_keeps_complete_prefix():
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    token = CancellationToken()
    load = np.zeros(54)
    load[6*8+1] = .08
    names = []

    def after_case(case_id):
        names.append(case_id)
        if len(names) == 3:
            token.cancel("bounded panel prefix")

    result = analysis.solve_linear_many(
        tuple((f"case-{index}", load) for index in range(10)),
        selected_dofs=(6*8+1,), cancellation_token=token,
        on_case=after_case)
    assert result.status == "cancelled"
    assert names == ["case-0", "case-1", "case-2"]
    assert tuple(case.case_id for case in result.cases) == tuple(names)
    assert result.factorization_count == 1
    assert result.solve_count == 3
    assert all(case.internal_strain_energy > 0. for case in result.cases)
    assert all(not case.point_multipliers and not case.point_gaps
               for case in result.cases)
    with pytest.raises(ValueError, match="selected physical B2 member"):
        analysis.solve_linear_many(
            (("bad", load),), selected_dofs=(0,),
            recover_members={"missing": (.5,)})
    assert analysis._handle.solve_count == 3
    analysis.close()


@pytest.mark.parametrize("close_after", (1, 3, 5))
@pytest.mark.parametrize("cancel_too", (False, True))
def test_point_panel_batch_callback_close_preserves_complete_prefix(close_after, cancel_too):
    nodes, shells, members, points, free = _panel()
    definition = stiffened_shell.B2Q4PanelDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54))
    analysis = stiffened_shell.create_analysis(definition=definition)
    control = stiffened_shell.create_analysis(definition=definition)
    load = np.zeros(54)
    load[6*8+1] = .08
    selected = (6*8+1, 6*5+2)
    recovery = {"beam-right": (.25, .75)}
    token = CancellationToken()
    names = []
    factor = weakref.ref(analysis._handle)

    def after_case(case_id):
        names.append(case_id)
        if len(names) == close_after:
            analysis.close()
            analysis.close()  # Closing remains idempotent at a callback boundary.
            assert factor() is None
            if cancel_too:
                token.cancel("closed complete prefix")

    try:
        expected = control.solve_linear(load, recover_members=recovery)
        assert expected.status == "completed", expected.failure_reason
        result = analysis.solve_linear_many(
            tuple((f"case-{index}", load) for index in range(5)),
            selected_dofs=selected, selected_points=("joint-right",),
            recover_members=recovery, cancellation_token=token, on_case=after_case)
        assert result.status == ("completed" if close_after == 5 else "cancelled")
        assert tuple(item.case_id for item in result.cases) == tuple(names)
        assert len(result.cases) == close_after
        assert result.factorization_count == 1 and result.solve_count == close_after
        assert result.maximum_rhs_chunk == 1 and not result.production_qualified
        if close_after < 5:
            assert result.failure_reason == "experimental panel closed after " + names[-1]
        else:
            assert result.failure_reason is None
        for case in result.cases:
            assert case.selected_displacements == tuple(expected.displacements[i] for i in selected)
            assert case.selected_affine_reactions == tuple(expected.affine_reactions[i] for i in selected)
            assert case.point_multipliers == (expected.point_multipliers[1],)
            assert case.point_gaps == (expected.point_gaps[1],)
            assert case.beam_resultants_local == expected.beam_resultants_local
            assert case.internal_strain_energy == expected.internal_strain_energy
        assert analysis._handle is None
        with pytest.raises(ExperimentalB2Q4PanelError, match="panel is closed"):
            analysis.solve_linear(load)
        with pytest.raises(ExperimentalB2Q4PanelError, match="panel is closed"):
            analysis.solve_linear_many((("later", load),), selected_dofs=selected)
    finally:
        analysis.close()
        control.close()


@pytest.mark.parametrize("failure_type", (RuntimeError, MemoryError))
def test_point_panel_batch_close_does_not_swallow_callback_error(failure_type):
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    load = np.zeros(54)
    load[6*8+1] = .08
    failure = failure_type("original callback failure")
    factor = weakref.ref(analysis._handle)

    def after_case(_case_id):
        assert analysis._handle.factorization_count == 1
        assert analysis._handle.solve_count == 1
        analysis.close()
        assert factor() is None
        raise failure

    try:
        with pytest.raises(failure_type) as caught:
            analysis.solve_linear_many(
                (("first", load), ("later", load)), selected_dofs=(6*8+1,),
                on_case=after_case)
        assert caught.value is failure
        # The preserved original traceback must not retain the closed factor.
        assert factor() is None
        assert analysis._handle is None
    finally:
        analysis.close()


def test_point_panel_batch_rejects_mutated_load_and_preserves_failed_prefix():
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    first = np.zeros(54)
    first[6*8+1] = .08
    second = first.copy()
    cases = (("first", first), ("second", second))

    def mutate_next(_case_id):
        second[6*8+1] = .16

    with pytest.raises(ValueError, match="changed after submission"):
        analysis.solve_linear_many(
            cases, selected_dofs=(6*8+1,), on_case=mutate_next)
    assert analysis._handle.factorization_count == 1
    assert analysis._handle.solve_count == 1

    original = analysis._handle

    class FailSecond:
        @property
        def factorization_count(self):
            return original.factorization_count

        @property
        def solve_count(self):
            return original.solve_count

        def solve(self, rhs):
            if original.solve_count == 2:
                raise RuntimeError("injected backend failure")
            return original.solve(rhs)

    analysis._handle = FailSecond()
    failed = analysis.solve_linear_many(
        (("accepted", first), ("broken", first), ("later", first)),
        selected_dofs=(6*8+1,))
    assert failed.status == "failed"
    assert tuple(item.case_id for item in failed.cases) == ("accepted",)
    assert failed.failure_reason == "broken: injected backend failure"
    assert failed.factorization_count == 1
    assert failed.solve_count == 2
    analysis.close()


@pytest.mark.parametrize("tip_dof", (1, 3))
def test_small_load_requires_relative_equilibrium_not_unit_floor(tip_dof):
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    load = np.zeros(54)
    load[6*8+tip_dof] = 1e-12
    reference = analysis.solve_linear(load*1e12)
    small = analysis.solve_linear(load)
    assert reference.status == small.status == "completed"
    np.testing.assert_allclose(small.displacements,
                               np.asarray(reference.displacements)*1e-12,
                               rtol=1e-8, atol=1e-24)

    # A backend returning zero for this small nonzero RHS used to pass the
    # former absolute 1 N residual floor and claim a completed solution.
    factor = analysis._handle
    analysis._handle = SimpleNamespace(
        solve=lambda rhs: np.zeros_like(rhs),
        factorization_count=factor.factorization_count,
        solve_count=factor.solve_count)
    failed = analysis.solve_linear(load)
    assert failed.status == "failed"
    assert "equilibrium" in failed.failure_reason
    analysis.close()


def test_mixed_force_scales_report_insufficient_resolution():
    nodes, shells, members, points, free = _panel()
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    load = np.zeros(54)
    load[6*2+2] = 1000.
    load[6*8+1] = 1e-12
    result = analysis.solve_linear(load)
    assert result.status == "failed"
    assert result.failure_reason == "panel mixed-load resolution insufficient"
    analysis.close()


@pytest.mark.parametrize("width,depth,prescribed", (
    (.012, .003, False), (.004, .001, False), (.012, .003, True),
))
def test_slender_physical_b2_panel_matches_independent_high_precision_system(
        width, depth, prescribed):
    """Mixed-system scaling restores a sound weak beam/strong shell solve."""
    nodes, shells, members, points, free = _panel()
    shells = tuple(replace(shell, E=210e9, thickness_m=.07)
                   for shell in shells)
    section = LinearSection.generalized(_physical_section(width, depth)[0])
    members = tuple(replace(member, section=section) for member in members)
    transformation = sparse.eye(54, format="lil")[:, free]
    if width == .012:
        # A full-rank, non-orthogonal affine basis must recover the same
        # physical solution after the mixed-system coordinate scaling.
        transformation[6*2, 0] = .25
    transformation = transformation.tocsr()
    u0 = np.zeros(54)
    if prescribed:
        u0[0] = 1e-10
        u0[6*8+2] = 2e-10
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            transformation, u0))
    load = np.zeros(54)
    load[6*8+1] = 100.
    load[6*8+3] = 10.
    result = analysis.solve_linear(load)
    assert result.status == "completed"
    assert result.factorization_count == result.solve_count == 1
    assert np.min(analysis._factor_scale) > 0.
    assert np.max(analysis._factor_scale)/np.min(analysis._factor_scale) > 1e3

    # Separate dense gather/scatter and a 100-digit solve of its stored
    # binary64 equations avoid using the owner's factorization as the oracle.
    stiffness, constraint = _independent_dense(
        nodes, shells, members, points)
    basis = transformation.toarray()
    reduced_stiffness = basis.T@stiffness@basis
    reduced_constraint = constraint@basis
    matrix = sparse.csr_matrix(np.block([
        [reduced_stiffness, reduced_constraint.T],
        [reduced_constraint, np.zeros((12, 12))],
    ]))
    oracle = _high_precision_rounded_matrix_solve(
        matrix, np.r_[basis.T@(load-stiffness@u0), -constraint@u0])
    expected = basis@oracle[:basis.shape[1]]+u0
    np.testing.assert_allclose(result.displacements, expected,
                               rtol=1e-8, atol=1e-12)
    multipliers = np.array([value for _, row in result.point_multipliers
                            for value in row])
    np.testing.assert_allclose(multipliers, oracle[basis.shape[1]:],
                               rtol=1e-8, atol=1e-9)
    displacement = np.asarray(result.displacements)
    assert np.max(np.abs(constraint@displacement)) < 1e-12
    # Recover ground/affine reactions from the independently assembled system.
    # The nonorthogonal affine basis and nonzero u0 are part of this same case.
    expected_reaction = (stiffness@expected+
                         constraint.T@oracle[basis.shape[1]:]-load)
    reaction = np.asarray(result.affine_reactions)
    np.testing.assert_allclose(reaction, expected_reaction,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(basis.T@reaction, 0., rtol=0., atol=1e-9)
    # K u + C^T lambda = f + r, C u = 0 and T^T r = 0 imply
    # 2 U = u.f + u0.r. Omitting prescribed-support work is incorrect.
    applied_work = float(displacement@load)
    prescribed_work = float(u0@reaction)
    work_scale = abs(applied_work)+abs(prescribed_work)
    assert work_scale > 0.
    assert abs(2*result.internal_strain_energy-
               applied_work-prescribed_work) <= 1e-8*work_scale
    if prescribed:
        assert abs(float(u0@expected_reaction)) > 0.
    analysis.close()


def test_prescribed_offset_only_rejects_a_zero_numerical_solution():
    nodes, shells, members, points, free = _panel()
    u0 = np.zeros(54)
    u0[0] = 1e-5
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            sparse.eye(54, format="csr")[:, free], u0))
    load = np.zeros(54)
    assert analysis.solve_linear(load).status == "completed"
    tiny = load.copy()
    tiny[6*8+1] = 1e-12
    underresolved = analysis.solve_linear(tiny)
    assert underresolved.status == "failed"
    assert underresolved.failure_reason == "panel mixed-load resolution insufficient"
    original = analysis._handle
    analysis._handle = SimpleNamespace(
        solve=lambda rhs: np.zeros_like(rhs),
        factorization_count=original.factorization_count,
        solve_count=original.solve_count)
    failed = analysis.solve_linear(load)
    assert failed.status == "failed"
    assert failed.failure_reason == "panel point gap or equilibrium failed"
    analysis.close()


def test_affine_work_projection_preserves_a_cancelled_explicit_load():
    nodes, shells, members, points, free = _panel()
    transformation = sparse.eye(54, format="lil")[:, free]
    # An admissible reduced translation has three physical contributions.
    # The stored binary64 dot must retain the 1 N work between the two
    # opposing 1e16 N terms, regardless of CSR accumulation order.
    transformation[free[1], 0] = 1.
    transformation[free[2], 0] = 1.
    analysis = stiffened_shell.create_analysis(definition=
        stiffened_shell.B2Q4PanelDefinition(
            stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
            nodes, shells, members, points,
            transformation.tocsr(), np.zeros(54)))
    load = np.zeros(54)
    load[free[0]] = 1e16
    load[free[1]] = 1.
    load[free[2]] = -1e16
    projected = analysis._project(load)
    assert projected[0] == 1.
    assert np.count_nonzero(projected[1:]) == 2
    # A normal-sized exact dot can be recovered after cancellation, but a
    # subnormal projected work row cannot meet the relative resolution rule.
    subnormal = np.zeros(54)
    subnormal[free[0]] = np.nextafter(0., 1.)
    with pytest.raises(_NumericalFailure, match="not representable"):
        analysis._project(subnormal)
    analysis.close()


def test_affine_projection_rejects_underresolved_subnormal_products():
    tiny = np.nextafter(0., 1.)
    for row in ((.3, .3), (.3,)):
        projection = sparse.csr_matrix(np.array([row]))
        owner = SimpleNamespace(
            _Tt=projection,
            _projection_multi=np.array([0] if len(row) > 1 else [], dtype=int),
            _projection_single=np.array([0] if len(row) == 1 else [], dtype=int))
        with pytest.raises(_NumericalFailure, match="not representable"):
            ExperimentalB2Q4PanelAnalysis._project(
                owner, np.full(len(row), tiny))


def test_disconnected_supported_panels_fail_topology_preflight():
    nodes, shells, members, points, _ = _panel()
    separated_nodes = {node_id: nodes[node_id] for node_id in (1, 2, 4, 5)}
    separated_nodes.update({
        7: (4., 0., 0.), 8: (5., 0., 0.),
        9: (5., 1., 0.), 10: (4., 1., 0.),
    })
    separated_shells = (
        shells[0], replace(shells[1], shell_id="far", node_ids=(7, 8, 9, 10)))
    separated_members = (
        members[0], replace(members[1], node_ids=(20, 21),
                            reference_points_m=((4.2, .5, .2),
                                                (5.2, .5, .2))))
    separated_points = (
        points[0], replace(points[1], shell_id="far"))
    with pytest.raises(ValueError, match="connected shell/beam point panel"):
        stiffened_shell.create_analysis(definition=
            stiffened_shell.B2Q4PanelDefinition(
                stiffened_shell.B2_E4_PL_FAST_LINEAR_V1,
                separated_nodes, separated_shells, separated_members,
                separated_points, sparse.eye(72, format="csr"),
                np.zeros(72)))
