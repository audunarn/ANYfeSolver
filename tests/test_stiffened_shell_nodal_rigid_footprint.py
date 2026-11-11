"""Selected-node footprint mechanics/lifecycle, not F04/M4 qualification."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver._stiffened_shell_rotation_charts import physical_frames
from anyfesolver._stiffened_shell_nodal_footprint import selected_wrenches
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _q4_shell_j2_panel
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition


def _definition(mixed=False, *, selected_only=False):
    base = _panel()
    right = ((2, 3, 6), (2, 6, 5)) if mixed else ((2, 3, 6, 5),)
    shell = replace(base.shell, q4_elements=((1, 2, 5, 4),) if mixed else base.shell.q4_elements,
                    s3_elements=right if mixed else ())
    if mixed == "all-s3":
        shell = replace(shell, q4_elements=(), s3_elements=((1, 2, 5), (1, 5, 4))+right)
    footprint = ss.NodalRigidFootprint("tip-pad", (2, 3, 6, 5), 105,
                                     () if selected_only else right)
    return replace(base, shell=shell, points=(base.points[0],),
                   q4_pullback_policy=(ss.Q4_ALGORITHMIC_PULLBACK_V1 if shell.q4_elements
                                       else "historical_corotational"),
                   footprints=(footprint,))


def _owner(request, definition):
    owner = ss.create_analysis(definition=definition)
    request.addfinalizer(owner.close)
    return owner


def _cross(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_complete_patch_finite_rigid_fields_and_zero_objective_deformation(request, mixed):
    owner = _owner(request, _definition(mixed))
    rotation = Rotation.from_rotvec((.6, -.35, .4)); matrix = rotation.as_matrix()
    translation = np.array((.2, -.15, .3))
    shell_reference = np.asarray(owner._definition.shell.coordinates_m)
    shell_positions = rotation.apply(shell_reference)+translation
    for joint in owner._connections:
        positions = joint.reference_positions@matrix.T+translation
        frames = np.array([matrix@q for q in joint.reference_frames])
        trial = joint.evaluate(positions, frames, np.zeros(6))
        np.testing.assert_allclose(trial.constraints, 0., rtol=0., atol=1e-12)
    # Actual full-map local coordinates of every declared Q4/S3 cell vanish.
    for _, element, _, _ in owner._shell_parts:
        reference = np.array([owner._shell_model.mesh.nodes[node].coords() for node in element.node_ids])
        u = np.column_stack((rotation.apply(reference)+translation-reference,
                             np.tile(rotation.as_rotvec(), (len(reference), 1)))).ravel()
        local, _, _ = deformation(reference, u, stable_translation=True)
        np.testing.assert_allclose(local, 0., rtol=0., atol=1e-12)
        # Independent affine interpolation reproduces the same body field.
        weights = np.full(len(reference), 1./len(reference))
        np.testing.assert_allclose(weights@(reference+u.reshape(-1, 6)[:, :3]),
                                   rotation.apply(weights@reference)+translation, rtol=0., atol=1e-12)
    q_rigid = np.zeros((owner._K.shape[0], 6))
    for domain, ids, offset in (("shell", owner._shell_ids, 0),
                                ("beam", owner._beam_ids, owner._shell_count)):
        for i, node in enumerate(ids):
            reference = shell_reference[i] if domain == "shell" else owner._beam_nodes[node][0]
            q_rigid[offset+6*i:offset+6*i+3, :3] = np.eye(3)
            q_rigid[offset+6*i:offset+6*i+3, 3:] = -_cross(reference)
            q_rigid[offset+6*i+3:offset+6*i+6, 3:] = np.eye(3)
    np.testing.assert_allclose(owner._C@q_rigid, 0., rtol=0., atol=1e-12)
    assert owner._definition.footprints[0].descriptor()["semantics"] == "complete_selected_cell_patch"
    assert np.isfinite(shell_positions).all()


def test_three_selected_nodes_do_not_constrain_omitted_q4_vertex(request):
    base = _definition()
    group = ss.NodalRigidFootprint("three-nodes", (2, 3, 6), 105)
    owner = _owner(request, replace(base, footprints=(group,)))
    omitted = 6*owner._shell_index[5]+2
    q = np.zeros(owner._K.shape[0]); q[omitted] = .01
    np.testing.assert_array_equal(owner._C@q, 0.)
    assert owner._definition.footprints[0].descriptor()["semantics"] == "selected_nodes_only"


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_sparse_reference_rows_match_independent_full_pose_map(request, mixed):
    owner = _owner(request, _definition(mixed))
    expected = np.zeros(owner._C.shape)
    assert owner._point_incidence[:1] == owner._definition.points
    for i, point in enumerate(owner._point_incidence):
        shell = 6*owner._shell_index[point.shell_node]
        beam = owner._shell_count+6*owner._beam_index[point.beam_node]
        arm = owner._beam_nodes[point.beam_node][0]-owner._shell_xyz[owner._shell_index[point.shell_node]]
        rows = slice(6*i, 6*i+6)
        expected[rows, shell:shell+6] = np.block([[-np.eye(3), _cross(arm)], [np.zeros((3, 3)), -np.eye(3)]])
        expected[rows, beam:beam+6] = np.eye(6)
    np.testing.assert_allclose(owner._C.toarray(), expected, rtol=0., atol=1e-12)
    assert owner._C.shape[0] == 30 and owner._C.nnz <= 72*5
    assert owner._footprint_bindings[0][1] == (1, 2, 3, 4)
    # Explicit pair expansion is an implementation oracle for this discrete
    # law only. Its distinct grouping identity cannot accept group checkpoints.
    chain = _owner(request, replace(owner._definition, points=owner._point_incidence, footprints=()))
    np.testing.assert_array_equal(owner._C.toarray(), chain._C.toarray())
    np.testing.assert_array_equal(owner._K.toarray(), chain._K.toarray())
    np.testing.assert_array_equal(owner._A.toarray(), chain._A.toarray())
    assert owner.preflight()["identity"] != chain.preflight()["identity"]


def test_nonzero_multiplier_actual_left_update_derivative_and_work(request):
    owner = _owner(request, _definition())
    joint = owner._connections[1]
    positions = joint.reference_positions+np.array(((.02, -.01, .03), (-.01, .03, -.02)))
    frames = Rotation.from_rotvec(((.12, -.08, .05), (-.06, .1, .04))).as_matrix()
    multiplier = np.array((30., -20., 40., 8., -7., 6.))
    trial = joint.evaluate(positions, frames, multiplier)
    row_scale = np.array((100.,)*3+(200.,)*3+(100.,)*3+(200.,)*3+(2.,)*3+(1.,)*3)
    input_scale = np.array((2.,)*3+(1.,)*3+(2.,)*3+(1.,)*3+(100.,)*3+(200.,)*3)
    for direction in (np.sin(np.arange(18)+.3), np.cos(.7*np.arange(18)+.2)):
        direction = direction/np.linalg.norm(direction)*input_scale
        predicted = trial.spatial_jacobian@direction
        errors = []
        work_errors = []
        for h in (1e-5, 3e-6, 1e-6):
            evaluations = []
            for sign in (1., -1.):
                delta = sign*h*direction
                x = positions+delta[:12].reshape(2, 6)[:, :3]
                q = Rotation.from_rotvec(delta[:12].reshape(2, 6)[:, 3:]).as_matrix()@frames
                evaluations.append(joint.evaluate(x, q, multiplier+delta[12:]))
            derivative = (evaluations[0].residual-evaluations[1].residual)/(2.*h)
            errors.append(float(np.linalg.norm((derivative-predicted)/row_scale)/np.linalg.norm(predicted/row_scale)))
            work = (evaluations[0].potential-evaluations[1].potential)/(2.*h)
            work_errors.append(abs(work-trial.residual@direction)/max(1., abs(trial.residual@direction)))
        assert all(np.isfinite(errors)) and max(errors) <= 1e-6
        assert max(work_errors) <= 1e-6


def _group_checks(owner, result):
    selection = ss.NodalRigidFootprintSelection(("tip-pad",))
    group, = owner.recover(selection=selection)
    assert type(group) is ss.NodalRigidFootprintWrench and not group.production_qualified
    assert group.generation == result.generation
    assert group.shell_node_ids == (2, 3, 6, 5)
    assert tuple(p.point_index for p in group.point_wrenches) == (1, 2, 3, 4)
    assert tuple(p.shell_node_id for p in group.point_wrenches) == group.shell_node_ids
    np.testing.assert_allclose((np.array(group.shell_action)+group.beam_action)/
                               np.array((10.,)*3+(20.,)*3), 0., rtol=0., atol=1e-8)
    assert group.configuration == ("current" if hasattr(result, "cell_frames") else "reference_linear")
    return group


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_linear_physical_wrench_reference_restart_and_selection(request, mixed):
    definition = _definition(mixed); owner = _owner(request, definition)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 1., 0.)),)
    result = owner.solve_linear(actions); group = _group_checks(owner, result)
    assert owner.recover() is result
    saved = owner.checkpoint()
    resumed = _owner(request, definition); resumed.restore(saved)
    assert resumed.checkpoint() == saved
    assert resumed.recover(selection=ss.NodalRigidFootprintSelection(("tip-pad",))) == (group,)
    with pytest.raises(ValueError, match="unknown"):
        owner.recover(selection=ss.NodalRigidFootprintSelection(("missing",)))
    with pytest.raises(ValueError, match="typed"):
        owner.recover(selection=("tip-pad",))
    assert owner.checkpoint() == saved
    # Each group action is the actual transpose action, not raw material lambda.
    rows = owner._C[6:30]
    exerted = -(rows.T@result.multipliers[6:30])
    beam_slot = owner._shell_count+6*owner._beam_index[105]
    np.testing.assert_allclose(exerted[beam_slot:beam_slot+6], group.beam_action, rtol=0., atol=1e-12)


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_finite_eccentric_group_balance_restart_without_rebase(request, mixed):
    definition = _definition(mixed); owner = _owner(request, definition)
    with pytest.raises(ValueError, match="translation forces only"):
        owner.solve((ss.NodalAction("beam", 105, (0., 0., 10., 0., 1., 0.)),), (.25,))
    assert owner._generation == 0 and owner._finite_driver is None
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    prefix = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    _group_checks(owner, prefix); saved = owner.checkpoint()
    control = _owner(request, definition); control.restore(saved)
    assert control.checkpoint() == saved
    successors = [branch.solve(actions, (1.,), max_iterations=20, max_backtracks=12)
                  for branch in (owner, control)]
    reference = successors[0]
    for branch, result in zip((owner, control), successors):
        _group_checks(branch, result)
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        assert result.accepted_factors == reference.accepted_factors
        np.testing.assert_allclose(result.beam_positions/2., reference.beam_positions/2., rtol=0., atol=1e-9)
        for field in ("beam_frames", "cell_frames"):
            np.testing.assert_allclose(getattr(result, field), getattr(reference, field), rtol=0., atol=1e-9)
        scale = np.array((10.,)*6+(20.,)*12)
        np.testing.assert_allclose(result.cell_resultants/scale, reference.cell_resultants/scale, rtol=0., atol=1e-9)
        for field in ("multipliers", "support_reactions"):
            np.testing.assert_allclose(getattr(result, field).reshape(-1, 6)/np.array((10.,)*3+(20.,)*3),
                                       getattr(reference, field).reshape(-1, 6)/np.array((10.,)*3+(20.,)*3), rtol=0., atol=1e-9)
        q = branch._finite_driver._state.q
        reference_state = owner._finite_driver._state
        np.testing.assert_allclose(q[:branch._shell_count].reshape(-1, 6)[:, :3]/2.,
                                   reference_state.q[:owner._shell_count].reshape(-1, 6)[:, :3]/2., rtol=0., atol=1e-9)
        np.testing.assert_allclose(physical_frames(q[:branch._shell_count], branch._finite_driver._state.shell_anchors),
                                   physical_frames(reference_state.q[:owner._shell_count], reference_state.shell_anchors), rtol=0., atol=1e-9)
        positions = np.vstack((branch._shell_xyz+q[:branch._shell_count].reshape(-1, 6)[:, :3], result.beam_positions))
        support = result.support_reactions[:6*len(positions)].reshape(-1, 6)
        force = support[:, :3].sum(0)+np.array((0., 0., 10.))
        moment = (support[:, 3:]+np.cross(positions, support[:, :3])).sum(0)+np.cross(positions[-1], (0., 0., 10.))
        np.testing.assert_allclose(force/10., 0., rtol=0., atol=1e-8)
        np.testing.assert_allclose(moment/20., 0., rtol=0., atol=1e-8)


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_cancel_and_injected_plastic_candidate_preserve_elastic_prefix(request, monkeypatch, mixed):
    owner = _owner(request, _definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    control = ss.AdaptiveIncrementControl(4, 64)
    prefix = owner.solve(actions, (.25,), increment_control=control, max_iterations=20, max_backtracks=12)
    saved = owner.checkpoint(); generation = prefix.generation
    original = finite.cancellation_safe_point
    original_checkpoint = finite.FinitePointDriver._prospective_checkpoint
    for cancel_stage in ("stiffened-shell.finite.publish", "stiffened-shell.finite.controller-publish"):
        token = CancellationToken(); notifications = []; prepared = []
        def cancel(value, stage):
            if stage == cancel_stage:token.cancel("footprint before commit")
            return original(value, stage)
        def prospective(driver, *args, **kwargs):
            raw = original_checkpoint(driver, *args, **kwargs); prepared.append(raw)
            return raw
        with monkeypatch.context() as patch:
            patch.setattr(finite, "cancellation_safe_point", cancel)
            patch.setattr(finite.FinitePointDriver, "_prospective_checkpoint", prospective)
            with pytest.raises(SolveCancelled, match="before commit"):
                owner.solve(actions, (.5,), increment_control=control, cancellation_token=token,
                            on_accepted=notifications.append, max_iterations=20, max_backtracks=12)
        assert not notifications
        if cancel_stage.endswith("controller-publish"):assert prepared and all(prepared)
        assert owner.checkpoint() == saved and owner.recover() is prefix and owner._generation == generation
    operator_type = type(owner._beam_parts[0][0].operator)
    original_evaluate = operator_type.evaluate
    for history_field in ("accumulated", "plastic"):
        def yielded(self, *args, **kwargs):
            response = original_evaluate(self, *args, **kwargs)
            stations = response.history.stations
            change = ({"accumulated": (1e-12, 0.)} if history_field == "accumulated"
                      else {"plastic": ((1e-12, 0.),)+stations[0].plastic[1:]})
            station = replace(stations[0], **change)
            return replace(response, history=replace(response.history, stations=(station,)+stations[1:]))
        with monkeypatch.context() as patch:
            patch.setattr(operator_type, "evaluate", yielded)
            with pytest.raises(ValueError, match="elastic beam response only"):
                owner.solve(actions, (.5,), increment_control=control, max_iterations=20, max_backtracks=12)
        assert owner.checkpoint() == saved and owner.recover() is prefix and owner._generation == generation


def test_owned_definition_descriptor_and_selection(request):
    nodes = [2, 3, 6, 5]; cells = [[2, 3, 6, 5]]; names = ["tip-pad"]
    group = ss.NodalRigidFootprint("tip-pad", nodes, 105, cells)
    selection = ss.NodalRigidFootprintSelection(names)
    definition = replace(_definition(), footprints=[group])
    owner = _owner(request, definition); identity = owner.preflight()["identity"]
    nodes.append(1); cells[0][0] = 1; names.append("other"); definition.footprints.clear()
    descriptor = group.descriptor(); descriptor["beam_node"] = 101
    assert owner.preflight()["identity"] == identity and owner._definition.footprints == (group,)
    assert group.shell_nodes == (2, 3, 6, 5) and group.patch_cells == ((2, 3, 6, 5),)
    payload = json.loads(json.dumps(group.descriptor()))
    assert ss.NodalRigidFootprint(payload["footprint_id"], payload["shell_nodes"],
                                 payload["beam_node"], payload["patch_cells"]) == group
    assert selection.footprint_ids == ("tip-pad",)
    with pytest.raises(FrozenInstanceError):group.beam_node = 101


@pytest.mark.parametrize("case, message", (
    ("historical", "full-map Q4"), ("unknown", "known shell"),
    ("station", "nodal station"), ("collinear", "noncollinear"),
    ("missing-corner", "every cell vertex"), ("duplicate-id", "duplicate"),
    ("point-overlap", "overlapping"), ("group-overlap", "overlapping"),
    ("supported-dependent", "independent supported attachment rows"),
    ("shell-j2", "elastic shell"), ("fibre", "elastic shell"),
    ("weak-line", "point-owner-only"),
))
def test_explicit_negative_admission(request, case, message):
    base = _definition(); group = base.footprints[0]
    if case == "historical":bad = replace(base, q4_pullback_policy="historical_corotational")
    elif case == "unknown":bad = replace(base, footprints=(replace(group, shell_nodes=(2, 3, 999), patch_cells=()),))
    elif case == "station":bad = replace(base, footprints=(replace(group, beam_node=999),))
    elif case == "collinear":bad = replace(base, points=(), footprints=(replace(group, shell_nodes=(1, 2, 3), patch_cells=()),))
    elif case == "missing-corner":bad = replace(base, footprints=(replace(group, shell_nodes=(2, 3, 6)),))
    elif case == "duplicate-id":bad = replace(base, footprints=(group, group))
    elif case == "point-overlap":bad = replace(base, points=base.points+(ss.PointConnection(2, 103),))
    elif case == "group-overlap":bad = replace(base, footprints=(group, replace(group, footprint_id="other", beam_node=103)))
    elif case == "supported-dependent":bad = replace(base, fixed=base.fixed+tuple(ss.FixedCoordinate("shell", node, a) for node in group.shell_nodes for a in range(6)))
    elif case == "shell-j2":bad = replace(base, shell=_q4_shell_j2_panel().shell)
    elif case == "weak-line":bad = replace(base, lines=(ss.ExperimentalBL5Line(
        (1, 2, 3), ((0, 0), (0, 1), (1, 0), (1, 1))),))
    else:bad = replace(base, beams=_yielding_definition(False).beams)
    with pytest.raises(ValueError, match=message):_owner(request, bad)


def test_complete_patch_connectivity_is_explicit_policy(request):
    base = _definition()
    shell = replace(base.shell, node_ids=base.shell.node_ids+(7, 8),
                    coordinates_m=base.shell.coordinates_m+((3., -.5, 0.), (3., .5, 0.)),
                    q4_elements=base.shell.q4_elements+((3, 7, 8, 6),))
    group = ss.NodalRigidFootprint("separate-cells", (1, 2, 5, 4, 3, 7, 8, 6), 105,
                                 ((1, 2, 5, 4), (3, 7, 8, 6)))
    with pytest.raises(ValueError, match="connected complete"):
        _owner(request, replace(base, shell=shell, points=(), footprints=(group,)))


def test_diagonal_shared_vertices_and_reordered_declaration_do_not_invent_patch_edges(request):
    base = _definition()
    shell = replace(base.shell, coordinates_m=((0., 0., 0.), (1., 0., 0.), (1., 1., 0.),
                                              (0., 1., 0.), (2., 0., 0.), (-1., 1., 0.)),
                    q4_elements=((1, 2, 3, 4), (1, 5, 3, 6)))
    group = ss.NodalRigidFootprint("diagonal-touch", (1, 2, 3, 4, 5, 6), 105,
                                 ((1, 2, 3, 4), (1, 3, 5, 6)))
    with pytest.raises(ValueError, match="connected complete"):
        _owner(request, replace(base, shell=shell, points=(), footprints=(group,)))


@pytest.mark.parametrize("overflow", (False, True))
def test_group_recovery_retains_low_arm_moment_and_rejects_nonfinite_resultants(overflow):
    footprint = ss.NodalRigidFootprint("low-arm", (1, 2, 3), 101)
    force = 1e308 if overflow else 2.**54
    records = []
    for i in range(3):
        action = (0., force if i == 0 or overflow else 0., 0., 0., 0., -force if i == 0 else 0.)
        records.append(ss.PointTransferWrench(i, i+1, 101, "reference_linear",
            (1., 0., 0.), (0., 0., 0.), (2.**-54 if i == 0 else 0., 0., 0.), (0., 0., 0.),
            action, (0., -action[1], 0., 0., 0., -1. if i == 0 else 0.), (0.,)*6))
    owner = SimpleNamespace(_footprint_bindings=((footprint, (0, 1, 2)),))
    result = SimpleNamespace(point_wrenches=tuple(records), generation=1)
    selection = ss.NodalRigidFootprintSelection(("low-arm",))
    if overflow:
        with pytest.raises(ValueError, match="finite physical"):selected_wrenches(owner, result, selection)
    else:
        recovered, = selected_wrenches(owner, result, selection)
        assert recovered.shell_action[5] == 1. and recovered.beam_action[5] == -1.


@pytest.mark.parametrize("change", ("group", "order", "station", "chain"))
def test_checkpoint_grouping_membership_station_mismatch_rejected(request, change):
    base = _definition(); owner = _owner(request, base)
    owner.solve_linear((ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),))
    saved = owner.checkpoint(); group = base.footprints[0]
    if change == "group":definition = replace(base, footprints=(replace(group, footprint_id="other"),))
    elif change == "order":definition = replace(base, footprints=(replace(group, shell_nodes=tuple(reversed(group.shell_nodes))),))
    elif change == "station":definition = replace(base, footprints=(replace(group, beam_node=103),))
    else:definition = replace(base, points=owner._point_incidence, footprints=())
    other = _owner(request, definition)
    assert other.preflight()["identity"] != owner.preflight()["identity"]
    with pytest.raises(ValueError, match="incompatible"):other.restore(saved)
    assert other._generation == 0 and other._accepted is None and other._finite_driver is None


def test_empty_footprints_preserve_legacy_identity_recovery_and_json_checkpoint(request):
    base = _panel(); owner = _owner(request, base)
    result = owner.solve_linear((ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),))
    assert owner.recover() is result and owner.capabilities["attachment"] == ("reference point pose",)
    assert "nodal_rigid_footprints" not in owner.preflight()
    identity = sha256(json.dumps(dict(profile=base.profile, shell_ids=base.shell.node_ids,
        shell_xyz=np.asarray(base.shell.coordinates_m).tolist(), shell_cells=base.shell.q4_elements,
        shell_s3_cells=base.shell.s3_elements, thickness=base.shell.thickness_m,
        E=base.shell.young_pa, nu=base.shell.poisson, density=base.shell.density_kg_m3,
        beams=[b.sha256 for b in base.beams], points=[(p.shell_node, p.beam_node) for p in base.points],
        fixed=[(s.domain, s.node, s.component) for s in base.fixed]),sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    assert owner.preflight()["identity"] == identity
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    load = owner._accepted[1]
    from anyfesolver._stiffened_shell_ge_experimental import CHECKPOINT
    expected = json.dumps(dict(schema=CHECKPOINT, model=identity, generation=result.generation,
        q=q.tolist(), multipliers=result.multipliers.tolist(), load=load.tolist()),
        sort_keys=True,separators=(",", ":"),allow_nan=False).encode("ascii")
    assert owner.checkpoint() == expected


@pytest.mark.parametrize("mixed", (True, "all-s3"))
def test_loaded_stable_s3_complete_assembly_actual_update_derivative(request, record_property, mixed):
    owner = _owner(request, _definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    accepted = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    driver = owner._finite_driver
    assert owner._footprint_stable_s3 and driver._stable_s3
    assert np.linalg.norm(accepted.multipliers) > 0.
    # Public footprint rebase is unsupported. Differentiate the existing
    # anchored element law using a detached, independently represented chart;
    # this never publishes an anchor/reset or claims accepted continuation.
    # The accepted material predecessors remain fixed in both private charts.
    all_errors = []
    for anchored in (False, True):
        state = driver._state; saved = owner.checkpoint(); published = owner.recover()
        if anchored:
            q = state.q.copy(); anchors = []
            for row in q[:owner._shell_count].reshape(-1, 6):
                vector = row[3:].copy(); theta = float(np.linalg.norm(vector)); cross = _cross(vector)
                if theta == 0.:made = np.eye(3)
                else:made = np.eye(3)+np.sin(theta)/theta*cross+(1.-np.cos(theta))/(theta*theta)*(cross@cross)
                anchors.append(made); row[3:] = 0.
            state = replace(state, q=q, shell_anchors=np.array(anchors))
        origins = (driver._shell_predecessors, driver._beam_predecessors)
        args = dict(pattern=driver._pattern)
        residual, tangent, *_ = driver._assemble(state, .5, *origins, **args)
        count = len(state.q); shell = owner._shell_count
        nodal = shell+6*len(owner._beam_ids)
        # SI coordinate scales: m, rad, N, Nm. Complementary resultant rows
        # have m/rad units; retained cell rotation equations have Nm units.
        input_scale = np.ones(count+len(state.multipliers))
        row_scale = np.ones_like(input_scale)
        input_scale[:nodal] = np.tile((2., 2., 2., 1., 1., 1.), nodal//6)
        row_scale[:nodal] = np.tile((100., 100., 100., 200., 200., 200.), nodal//6)
        for member in range(len(owner._beam_parts)):
            base = nodal+24*member
            input_scale[base:base+24] = (1.,)*6+(100.,)*6+(200.,)*12
            row_scale[base:base+24] = (200.,)*6+(2.,)*6+(1.,)*12
        input_scale[count:] = np.tile((100., 100., 100., 200., 200., 200.), len(owner._connections))
        row_scale[count:] = np.tile((2., 2., 2., 1., 1., 1.), len(owner._connections))
        roles = (slice(0, shell), slice(shell, nodal), slice(nodal, count), slice(count, None))
        for phase in (.3, 1.1):
            direction = input_scale*np.sin(np.arange(len(input_scale))+.4+phase)
            direction[owner._fixed] = 0.
            expected = tangent@direction/row_scale
            for epsilon in (1e-5, 3e-6, 1e-6):
                plus = driver._advance(state, direction, epsilon)
                minus = driver._advance(state, direction, -epsilon)
                rp = driver._assemble(plus, .5, *origins, **args)[0]
                rm = driver._assemble(minus, .5, *origins, **args)[0]
                actual = (rp-rm)/(2.*epsilon)/row_scale
                all_errors.append([float(np.linalg.norm((actual-expected)[role])/
                    max(1e-12, np.linalg.norm(expected[role]))) for role in roles])
        assert owner.checkpoint() == saved and owner.recover() is published
    record_property("loaded_actual_update_derivative_role_errors", json.dumps(all_errors))
    assert np.isfinite(all_errors).all() and np.max(all_errors) <= 1e-6


@pytest.mark.parametrize("mixed", (True, "all-s3"))
def test_stable_footprint_complete_generalized_history_and_atomic_corrupt_restore(request, mixed):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode, encode, MAGIC
    from anyfesolver._stiffened_shell_beam_history import shapes, histories
    from anyfesolver._stiffened_shell_nodal_footprint import (
        stable_s3_policy_descriptor, retained_scale_policy_descriptor,
        chart_admission_policy_descriptor)

    definition = _definition(mixed); owner = _owner(request, definition)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    accepted = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    saved = owner.checkpoint(); driver = owner._finite_driver; state = driver._state
    expected = (("q", state.q.shape), ("frames", state.frames.shape),
        ("cells", state.cells.shape), ("resultants", state.resultants.shape),
        ("multipliers", state.multipliers.shape), ("pattern", driver._pattern.shape),
        ("accepted_factors", (2,)))+shapes(owner._beam_parts, "origin")+shapes(owner._beam_parts, "accepted")
    kind, generation, arrays = decode(saved, owner._identity, lambda k, g: expected)
    assert kind == "finite_native_material_stable" and generation == 2
    for prefix, source in (("origin", driver._beam_predecessors), ("accepted", driver._beam_origins)):
        assert arrays[prefix+"_fibres"].shape == (0, 4)
        assert arrays[prefix+"_plastic"].shape[0] > 0
        assert histories(owner._beam_parts, arrays, prefix) == source
        assert not np.any(arrays[prefix+"_plastic"]) and not np.any(arrays[prefix+"_accumulated"])
    restored = _owner(request, definition); restored.restore(saved)
    assert restored.checkpoint() == saved
    assert owner.preflight()["footprint_finite_numerical_policy"] == stable_s3_policy_descriptor()

    # These all have correct framing/digests: restoration must validate policy,
    # complete layout and elastic history semantics before atomic publication.
    corrupt = []
    for prefix in ("origin", "accepted"):
        for field in ("plastic", "accumulated"):
            copied = {name: value.copy() for name, value in arrays.items()}
            target = copied[prefix+"_"+field]
            if field == "plastic":target[0, 0] = (1., 2.**-54)
            else:target[0] = (1., 2.**-54)
            corrupt.append((encode(kind, owner._identity, generation, tuple(copied.items())), "elastic beam response only"))
    corrupt.append((encode("finite_native_material", owner._identity, generation, tuple(arrays.items())), "numerical policy"))
    corrupt.append((encode(kind, owner._identity, generation, tuple(
        (name, value) for name, value in arrays.items() if name != "accepted_fibres")), "layout"))
    # Remove only the newly declared numerical-policy key to reconstruct the
    # provisional footprint identity, not an arbitrary corrupted byte digest.
    panel = definition.shell
    payload = dict(profile=definition.profile, shell_ids=panel.node_ids,
        shell_xyz=np.asarray(panel.coordinates_m).tolist(), shell_cells=panel.q4_elements,
        shell_s3_cells=panel.s3_elements, thickness=panel.thickness_m, E=panel.young_pa,
        nu=panel.poisson, density=panel.density_kg_m3, beams=[b.sha256 for b in definition.beams],
        points=[(p.shell_node, p.beam_node) for p in definition.points],
        fixed=[(v.domain, v.node, v.component) for v in definition.fixed],
        nodal_rigid_footprints=[g.descriptor() for g in definition.footprints],
        footprint_retained_scale_policy=retained_scale_policy_descriptor(),
        footprint_chart_admission_policy=chart_admission_policy_descriptor())
    if panel.q4_elements:payload["shell_q4_pullback_policy"] = definition.q4_pullback_policy
    provisional = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert provisional != owner._identity
    corrupt.append((encode(kind, provisional, generation, tuple(arrays.items())), "incompatible"))
    # Distinguish stale S3 arithmetic from stale retained scaling: this second
    # provisional identity includes stable S3, but lacks only the scale policy.
    payload["footprint_finite_numerical_policy"] = stable_s3_policy_descriptor()
    del payload["footprint_retained_scale_policy"]
    old_scale = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert old_scale != provisional and old_scale != owner._identity
    corrupt.append((encode(kind, old_scale, generation, tuple(arrays.items())), "incompatible"))
    payload["footprint_retained_scale_policy"] = retained_scale_policy_descriptor()
    del payload["footprint_chart_admission_policy"]
    old_chart = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert old_chart not in (old_scale, provisional, owner._identity)
    corrupt.append((encode(kind, old_chart, generation, tuple(arrays.items())), "incompatible"))
    for raw, message in corrupt:
        assert raw.startswith(MAGIC) and sha256(raw[:-32]).digest() == raw[-32:]
        target = _owner(request, definition)
        with pytest.raises(ValueError, match=message):target.restore(raw)
        assert target._generation == 0 and target._accepted is None and target._finite_driver is None
        assert owner.recover() is accepted and owner._generation == generation and owner.checkpoint() == saved


def test_stable_policy_is_absent_for_q4_footprint_and_empty_generalized_s3(request):
    q4 = _owner(request, _definition())
    empty_s3 = _owner(request, replace(_definition("all-s3"), footprints=()))
    assert not q4._footprint_stable_s3 and not empty_s3._footprint_stable_s3
    assert "footprint_finite_numerical_policy" not in q4.preflight()
    assert "footprint_finite_numerical_policy" not in empty_s3.preflight()
    assert "footprint_retained_scale_policy" in q4.preflight()
    assert "footprint_retained_scale_policy" not in empty_s3.preflight()


def test_unequal_section_scales_follow_retained_roles_and_work_conjugate_units():
    from anyfesolver import _stiffened_shell_ge_experimental as ge
    xyz = np.array(((0., 0., 0.), (2., 0., 0.)))
    nodes = {i: (np.array((float(i-1), 0., .1)), np.eye(3)) for i in (1, 2, 3)}
    elastic = np.diag((2., 3., 5., 7., 11., 13.))
    parts = ((SimpleNamespace(section=SimpleNamespace(elastic=elastic)), (), None),)
    length, shell, beam = ge._reference_scales(xyz, nodes, parts, 12, 42,
                                              footprint_retained_roles=True)
    old_length, old_shell, old_beam = ge._reference_scales(xyz, nodes, parts, 12, 42)
    force = np.array((2., 3., 5.)); moment = np.array((7., 11., 13.))*(1./length)
    expected = np.r_[np.tile(force, 2), np.tile(moment, 4)]
    np.testing.assert_array_equal(beam[24:], expected)
    np.testing.assert_array_equal(old_beam[24:], np.tile(np.r_[force, moment], 3))
    assert not np.array_equal(beam[24:], old_beam[24:])
    assert old_length == length
    np.testing.assert_array_equal(shell, old_shell)
    np.testing.assert_array_equal(beam[:24], old_beam[:24])

    scale = np.r_[shell, beam]; residual = np.zeros(54); applied = np.zeros(54)
    applied[12] = 2.; residual[12+24+3] = .01; residual[12+24+6] = .002
    floor = 1.*length  # 1N * reference length, in J.
    reference = ge._relative_equilibrium(residual, applied, scale, floor)
    assert reference != ge._relative_equilibrium(residual, applied, np.r_[old_shell, old_beam], floor)
    # Numeric reexpression m -> mm, N unchanged: work becomes Nmm. q=D*q_mm,
    # covectors become 1000*D.T*r, and the 1N*length floor also gains 1000.
    d = np.ones(54)
    d[:30] = np.tile((.001, .001, .001, 1., 1., 1.), 5)
    d[12+24+6:] = .001  # Native twelve moment coordinates; forces unchanged.
    converted_scale = scale/d
    converted_residual = 1000.*d*residual; converted_applied = 1000.*d*applied
    np.testing.assert_allclose(ge._relative_equilibrium(converted_residual,
        converted_applied, converted_scale, 1000.*floor), reference, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(ge._linear_equilibrium(converted_residual,
        converted_applied, converted_scale), ge._linear_equilibrium(residual, applied, scale), rtol=1e-14, atol=0.)
    assert converted_applied[12] == applied[12]  # N is unchanged.
    probe = np.sin(np.arange(54)+.3)
    np.testing.assert_allclose((probe/d)@converted_residual/1000., probe@residual, rtol=1e-14, atol=0.)

    # Canonical quiet-row controls independently distinguish the two packing
    # errors. L=2m, F*=100N, M*=2Nm, nodal applied force10N =>20J load work.
    correct = np.r_[(2., 2., 2., 1., 1., 1.), np.full(6, 100.), np.full(12, 2.)]
    historical = np.r_[(2., 2., 2., 1., 1., 1.), np.tile((100.,)*3+(2.,)*3, 3)]
    load = np.zeros(24); load[0] = 10.
    for slot, expected_old, expected_correct in ((9, 1e-9, 5e-8), (12, 5e-8, 1e-9)):
        quiet = np.zeros(24); quiet[slot] = 1e-8
        for criterion in (lambda s: ge._relative_equilibrium(quiet, load, s, 2.),
                          lambda s: ge._linear_equilibrium(quiet, load, s)):
            np.testing.assert_allclose(criterion(historical), expected_old, rtol=1e-14, atol=0.)
            np.testing.assert_allclose(criterion(correct), expected_correct, rtol=1e-14, atol=0.)


def test_q4_footprint_rejects_provisional_retained_scale_identity(request):
    from anyfesolver._stiffened_shell_nodal_footprint import (
        retained_scale_policy_descriptor, chart_admission_policy_descriptor)
    definition = _definition(); panel = definition.shell
    owner = _owner(request, definition)
    result = owner.solve_linear((ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),))
    saved = owner.checkpoint(); checkpoint = json.loads(saved)
    payload = dict(profile=definition.profile, shell_ids=panel.node_ids,
        shell_xyz=np.asarray(panel.coordinates_m).tolist(), shell_cells=panel.q4_elements,
        shell_s3_cells=panel.s3_elements, thickness=panel.thickness_m, E=panel.young_pa,
        nu=panel.poisson, density=panel.density_kg_m3, beams=[b.sha256 for b in definition.beams],
        points=[(p.shell_node, p.beam_node) for p in definition.points],
        fixed=[(v.domain, v.node, v.component) for v in definition.fixed],
        nodal_rigid_footprints=[g.descriptor() for g in definition.footprints],
        shell_q4_pullback_policy=definition.q4_pullback_policy,
        footprint_chart_admission_policy=chart_admission_policy_descriptor())
    old_scale = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["footprint_retained_scale_policy"] = retained_scale_policy_descriptor()
    assert sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == owner._identity
    assert old_scale != owner._identity
    checkpoint["model"] = old_scale
    raw = json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    target = _owner(request, definition)
    with pytest.raises(ValueError, match="incompatible"):target.restore(raw)
    assert target._generation == 0 and target._accepted is None and target._finite_driver is None
    assert owner.recover() is result and owner.checkpoint() == saved
    assert owner.preflight()["footprint_retained_scale_policy"] == retained_scale_policy_descriptor()
    assert owner.capabilities["footprint_retained_scale_policy"] == retained_scale_policy_descriptor()


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_footprint_rebase_and_current_model_chart_restore_reject_atomically(request, mixed):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import encode, encode_chart, decode_chart
    from anyfesolver._stiffened_shell_beam_history import arrays as history_arrays
    from anyfesolver._stiffened_shell_nodal_footprint import chart_admission_policy_descriptor
    definition = _definition(mixed); owner = _owner(request, definition)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    accepted = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    driver = owner._finite_driver; state = driver._state; saved = owner.checkpoint()
    captured = {name: getattr(state, name).copy() for name in ("q", "frames", "cells", "resultants", "multipliers")}
    histories = (driver._shell_origins, driver._beam_origins)
    with pytest.raises(ValueError, match="chart rebase is unsupported"):
        owner.rebase_rotations(expected_generation=accepted.generation)
    assert owner._generation == accepted.generation and owner.recover() is accepted
    assert driver._state is state and owner.checkpoint() == saved
    assert state.shell_anchors is None
    for name, value in captured.items():np.testing.assert_array_equal(getattr(state, name), value)
    assert driver._shell_origins is histories[0] and driver._beam_origins is histories[1]
    assert owner.preflight()["footprint_chart_admission_policy"] == chart_admission_policy_descriptor()
    assert owner.capabilities["footprint_chart_admission_policy"] == chart_admission_policy_descriptor()

    # Correctly framed chart bytes use the CURRENT model identity; unsupported
    # admission must not be bypassed by an externally constructed checkpoint.
    anchors = physical_frames(state.q[:owner._shell_count])
    q = state.q.copy(); q[:owner._shell_count].reshape(-1, 6)[:, 3:] = 0.
    arrays = (("q", q), ("frames", state.frames), ("cells", state.cells),
        ("resultants", state.resultants), ("multipliers", state.multipliers),
        ("pattern", driver._pattern), ("accepted_factors", np.array((.25, .5))))
    kind = "finite_chart"
    if owner._footprint_stable_s3:
        kind = "finite_chart_native_material_stable"
        arrays += history_arrays(owner._beam_parts, driver._beam_predecessors, "origin")
        arrays += history_arrays(owner._beam_parts, driver._beam_origins, "accepted")
    inner = encode(kind, owner._identity, 2, arrays)
    raw = encode_chart(owner._identity, 3, anchors, anchors, inner)
    assert sha256(raw[:-32]).digest() == raw[-32:]
    event_generation, decoded_anchors, witness, decoded_inner = decode_chart(raw, owner._identity, len(owner._shell_ids))
    assert event_generation == 3 and decoded_inner == inner
    np.testing.assert_array_equal(decoded_anchors, anchors); np.testing.assert_array_equal(witness, anchors)
    target = _owner(request, definition)
    with pytest.raises(ValueError, match="chart checkpoint restore is unsupported"):target.restore(raw)
    assert target._generation == 0 and target._accepted is None and target._finite_driver is None
    assert owner.checkpoint() == saved and owner.recover() is accepted
