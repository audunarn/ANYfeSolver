"""Experimental tilted Q4/S3 point owner; not curved-shell qualification."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell as ss
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_variational_shell import deformation, response
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from test_stiffened_shell_curved_reference_seam import curved_panel


def curved_triangles(*, mixed=False):
    base = curved_panel()
    quads = base.shell.q4_elements
    selected = quads[1:] if mixed else quads
    triangles = tuple(cell for a, b, c, d in selected
                      for cell in ((a, b, c), (a, c, d)))
    shell = replace(base.shell,
                    q4_elements=quads[:1] if mixed else (),
                    s3_elements=triangles)
    return replace(base, shell=shell,
                   shell_reference_policy=ss.CURVED_Q4_S3_POINT_REFERENCE_V1)


def _independent_accepted_point_work(owner, result):
    """Check physical point work without calling the production joint map."""
    state = owner._finite_driver._state
    directions = np.eye(12)
    assert len(result.point_wrenches) == len(owner._connections) == 3
    for index, (joint, record) in enumerate(zip(owner._connections,
                                                 result.point_wrenches)):
        shell_node = owner._shell_index[record.shell_node_id]
        beam_node = owner._beam_index[record.beam_node_id]
        shell_anchor = (np.eye(3) if state.shell_anchors is None else
                        state.shell_anchors[shell_node])
        shell_rotation = Rotation.from_rotvec(
            state.q[6*shell_node+3:6*shell_node+6]).as_matrix()@shell_anchor
        beam_rotation = state.frames[beam_node]
        shell_position = np.asarray(record.shell_position_m) + record.shell_position_low_m
        beam_position = np.asarray(record.beam_position_m) + record.beam_position_low_m
        reference_shell, reference_beam = joint.reference_positions
        reference_shell_rotation, reference_beam_rotation = joint.reference_frames
        offset = reference_shell_rotation.T@(reference_beam-reference_shell)
        relative = reference_shell_rotation.T@reference_beam_rotation
        multiplier = state.multipliers[6*index:6*index+6]

        def potential(variation):
            varied = variation.reshape(2, 6)
            xs = shell_position+varied[0, :3]
            xb = beam_position+varied[1, :3]
            rs = Rotation.from_rotvec(varied[0, 3:]).as_matrix()@shell_rotation
            rb = Rotation.from_rotvec(varied[1, 3:]).as_matrix()@beam_rotation
            gap = np.r_[rs.T@(xb-xs)-offset,
                        Rotation.from_matrix(relative.T@rs.T@rb).as_rotvec()]
            return float(multiplier@gap), gap

        zero_work, zero_gap = potential(np.zeros(12))
        assert abs(zero_work) <= 1e-9*max(1.,np.linalg.norm(multiplier))
        np.testing.assert_allclose(zero_gap, record.gap, rtol=0., atol=1e-10)
        shell_action = np.asarray(record.shell_action)
        beam_action = np.asarray(record.beam_action)
        force_scale = max(1.,np.linalg.norm(shell_action),np.linalg.norm(beam_action))
        moment_scale = force_scale*max(1.,np.linalg.norm(shell_position),
                                       np.linalg.norm(beam_position))
        assert np.linalg.norm(shell_action[:3]+beam_action[:3])/force_scale <= 1e-8
        moment = (shell_action[3:]+np.cross(shell_position,shell_action[:3])+
                  beam_action[3:]+np.cross(beam_position,beam_action[:3]))
        assert np.linalg.norm(moment)/moment_scale <= 1e-8
        step = 1e-6
        work_scale = max(1.,np.linalg.norm(np.r_[shell_action,beam_action]))
        for direction in directions:
            work_derivative = (potential(step*direction)[0]-
                               potential(-step*direction)[0])/(2*step)
            recovered = -np.r_[shell_action,beam_action]@direction
            assert abs(work_derivative-recovered)/work_scale <= 1e-8


@pytest.mark.parametrize("mixed", (False, True))
def test_curved_s3_point_reference_linear_finite_restart(request, mixed):
    definition = curved_triangles(mixed=mixed)

    def make():
        owner = ss.create_analysis(definition=definition)
        request.addfinalizer(owner.close)
        return owner

    linear, owner, restored = make(), make(), make()
    assert "Q4/S3" in owner.capabilities["shell"]
    preflight = owner.preflight()
    assert not preflight["production_qualified"]
    assert preflight["stress_free_scaled_max"] <= 1e-10
    action = (ss.NodalAction("beam", 105, (0., 0., -100., 0., 0., 0.)),)
    reference = linear.solve_linear(action)
    assert reference.status == "completed"
    assert reference.free_equilibrium_ratio <= 1e-8
    assert reference.constraint_max <= 1e-9
    supported = np.zeros_like(reference.reactions)
    supported[linear._fixed] = reference.reactions[linear._fixed]
    np.testing.assert_allclose(
        supported[:linear._shell_count].reshape(-1, 6)[:, :3].sum(axis=0),
        (0., 0., 100.), rtol=1e-8, atol=1e-6)

    try:
        first = owner.solve(action, (.1, .2))
    except ss.FiniteIncrementFailure as error:
        pytest.fail(f"curved S3 finite candidate rejected at generation "
                    f"{owner._generation}: {asdict(error.attempt)}")
    assert first.free_equilibrium_ratio <= 1e-8
    assert first.constraint_max <= 1e-9
    _independent_accepted_point_work(owner, first)
    raw = owner.checkpoint()
    damaged = bytearray(raw)
    damaged[-1] ^= 1
    refused = make()
    with pytest.raises(ValueError):
        refused.restore(bytes(damaged))
    assert refused._generation == 0 and refused._finite_driver is None
    positions = np.array(definition.shell.coordinates_m, copy=True)
    positions[5, 2] += .002
    changed = replace(definition, shell=replace(definition.shell,
        coordinates_m=tuple(map(tuple, positions))))
    wrong = ss.create_analysis(definition=changed)
    request.addfinalizer(wrong.close)
    with pytest.raises(ValueError, match="incompatible|model"):
        wrong.restore(raw)
    assert wrong._generation == 0 and wrong._finite_driver is None
    restored.restore(raw)
    assert restored.checkpoint() == raw
    continued = owner.solve(action, (.3,))
    replay = restored.solve(action, (.3,))
    np.testing.assert_array_equal(continued.shell_displacement,
                                  replay.shell_displacement)
    np.testing.assert_array_equal(continued.cell_resultants,
                                  replay.cell_resultants)
    assert owner.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize("mixed", (False, True))
def test_curved_s3_point_mixed_translation_work(request, mixed):
    owner = ss.create_analysis(definition=curved_triangles(mixed=mixed))
    request.addfinalizer(owner.close)
    action = (ss.NodalAction("beam", 105, (30., -40., -100., 0., 0., 0.)),)
    accepted = owner.solve(action, (.1,))
    assert accepted.status == "completed"
    assert accepted.free_equilibrium_ratio <= 1e-8
    assert accepted.constraint_max <= 1e-9
    _independent_accepted_point_work(owner, accepted)


def test_curved_s3_policy_is_explicit_and_rejects_degenerate_reference():
    definition = curved_triangles()
    with pytest.raises(ValueError, match="curved reference"):
        ss.create_analysis(definition=replace(
            definition, shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1))
    with pytest.raises(ValueError, match="selected elastic Q4 or Q4/S3"):
        ss.create_analysis(definition=replace(
            definition, shell=replace(definition.shell, s3_elements=(),
                                     q4_elements=curved_panel().shell.q4_elements)))
    points = list(definition.shell.coordinates_m)
    points[4] = points[0]
    with pytest.raises(ValueError, match="degenerate|nondegenerate"):
        ss.create_analysis(definition=replace(
            definition, shell=replace(definition.shell,
                                      coordinates_m=tuple(points))))
    triangles = list(definition.shell.s3_elements)
    a, b, c = triangles[-1]
    triangles[-1] = (a, c, b)
    with pytest.raises(ValueError, match="oriented"):
        ss.create_analysis(definition=replace(
            definition, shell=replace(definition.shell,
                                      s3_elements=tuple(triangles))))


def test_curved_s3_no_change_stress_free_reference_keeps_identity():
    definition = curved_triangles()
    result = ss.create_stress_free_point_reference(definition, ())
    assert result.definition.shell_reference_policy == ss.CURVED_Q4_S3_POINT_REFERENCE_V1
    assert result.source_identity == result.target_identity
    assert result.stress_free_scaled_max <= 1e-10


@pytest.mark.parametrize("mixed", (False, True))
def test_curved_s3_changed_stress_free_reference_has_no_initial_gap(request, mixed):
    definition = curved_triangles(mixed=mixed)
    perturbation = (ss.ReferencePosePerturbation("shell", 6, (0., 0., .002)),)
    result = ss.create_stress_free_point_reference(definition, perturbation)
    assert result.source_identity != result.target_identity
    assert result.definition.shell_reference_policy == ss.CURVED_Q4_S3_POINT_REFERENCE_V1
    assert result.stress_free_scaled_max <= 1e-10
    owner = ss.create_analysis(definition=result.definition)
    request.addfinalizer(owner.close)
    assert owner.preflight()["stress_free_scaled_max"] <= 1e-10
    assert not np.array_equal(definition.shell.coordinates_m[5],
                              result.definition.shell.coordinates_m[5])
    action = (ss.NodalAction("beam", 105, (30., -40., -100., 0., 0., 0.)),)
    linear = ss.create_analysis(definition=result.definition)
    request.addfinalizer(linear.close)
    reference = linear.solve_linear(action)
    assert reference.status == "completed"
    assert reference.free_equilibrium_ratio <= 1e-8
    assert reference.constraint_max <= 1e-9
    first = owner.solve(action, (.1,))
    assert first.status == "completed"
    assert first.free_equilibrium_ratio <= 1e-8
    assert first.constraint_max <= 1e-9
    _independent_accepted_point_work(owner, first)
    checkpoint = owner.checkpoint()
    restored = ss.create_analysis(definition=result.definition)
    request.addfinalizer(restored.close)
    restored.restore(checkpoint)
    assert restored.checkpoint() == checkpoint
    continued = owner.solve(action, (.2,))
    replayed = restored.solve(action, (.2,))
    np.testing.assert_array_equal(continued.shell_displacement,
                                  replayed.shell_displacement)
    np.testing.assert_array_equal(continued.cell_resultants,
                                  replayed.cell_resultants)
    assert owner.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize("mixed", (False, True))
def test_curved_s3_common_rotated_reference_preserves_loaded_response(request, mixed):
    definition = curved_triangles(mixed=mixed)
    motion = Rotation.from_rotvec((.2, -.1, .3))
    matrix = motion.as_matrix()
    translation = np.array((.01, -.02, .03))
    rotation = tuple(motion.as_rotvec())
    rows = tuple(ss.ReferencePosePerturbation("shell", node,
        tuple(motion.apply(position)+translation-position), rotation)
        for node, position in zip(definition.shell.node_ids,
                                  definition.shell.coordinates_m))
    attached = {point.beam_node for point in definition.points}
    free_beam = {}
    for beam in definition.beams:
        element, _ = beam.instantiate()
        for node, position in zip(element.node_ids,
                                  element.operator.reference.coordinates):
            if node not in attached:
                free_beam[node] = position
    rows += tuple(ss.ReferencePosePerturbation("beam", node,
        tuple(motion.apply(position)+translation-position), rotation)
        for node, position in free_beam.items())
    moved = ss.create_stress_free_point_reference(definition, rows)
    assert moved.source_identity != moved.target_identity
    assert moved.stress_free_scaled_max <= 1e-10
    original = ss.create_analysis(definition=definition)
    target = ss.create_analysis(definition=moved.definition)
    request.addfinalizer(original.close)
    request.addfinalizer(target.close)
    for node in original._beam_ids:
        position, frame = original._beam_nodes[node]
        rotated_position, rotated_frame = target._beam_nodes[node]
        np.testing.assert_allclose(rotated_position,
                                   matrix@position+translation, atol=1e-14)
        np.testing.assert_allclose(rotated_frame, matrix@frame, atol=1e-14)

    force = np.array((30., -40., -100.))
    base_action = (ss.NodalAction("beam", 105, tuple(force)+(0., 0., 0.)),)
    rotated_action = (ss.NodalAction("beam", 105,
                      tuple(motion.apply(force))+(0., 0., 0.)),)
    base = original.solve_linear(base_action)
    transformed = target.solve_linear(rotated_action)
    assert base.status == transformed.status == "completed"
    assert max(base.free_equilibrium_ratio,
               transformed.free_equilibrium_ratio) <= 1e-8
    np.testing.assert_allclose(
        transformed.shell_displacement.reshape(-1, 6)[:, :3],
        base.shell_displacement.reshape(-1, 6)[:, :3]@matrix.T,
        rtol=1e-6, atol=1e-11)
    for old, new in zip(base.point_wrenches, transformed.point_wrenches):
        np.testing.assert_allclose(new.gap, old.gap, rtol=0., atol=1e-10)
        for name in ("shell_action", "beam_action"):
            original_action = np.asarray(getattr(old, name))
            rotated_action = np.asarray(getattr(new, name))
            np.testing.assert_allclose(rotated_action[:3],
                                       matrix@original_action[:3],
                                       rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(rotated_action[3:],
                                       matrix@original_action[3:],
                                       rtol=1e-6, atol=1e-7)


def test_tilted_s3_variational_tangent_and_energy_work():
    base = np.array(((0., 0., 0.), (1.1, 0., 0.), (.25, .95, 0.)))
    tilt = Rotation.from_rotvec((.2, -.3, .1))
    points = tilt.apply(base) + (.17, -.11, .08)
    normal = tilt.apply((0., 0., 1.))
    model = FEModel("curved-s3-variational-work")
    model.add_material("shell", 210e9, .3, density=7850.)
    for node, point in enumerate(points, 1):
        model.add_node(node, *map(float, point))
    element = create_shell_element(1, (1, 2, 3), "shell",
        formulation="e4-pl-s3-v2d", thickness=.018,
        reference_normal=tuple(normal))
    model.add_element(1, element)
    origin = element.init_model_bound_nonlinear_state(
        model.mesh, model.get_material("shell"), 3)
    zero = np.zeros(18)
    reference_force, _, _ = response(model, element, zero, origin, 3,
                                     stable_translation=True)
    assert np.linalg.norm(reference_force) < 1e-9

    q = .002*np.sin(np.arange(18)+.2)
    direction = np.cos(np.arange(18)+.3)
    direction /= np.linalg.norm(direction)
    force, tangent, _ = response(model, element, q, origin, 3,
                                 stable_translation=True)
    local_stiffness = element.compute_stiffness_matrix(
        model.mesh, model.get_material("shell"))

    def energy(coordinates):
        local = deformation(points, coordinates, stable_translation=True)[0]
        return .5*float(local@local_stiffness@local)

    chart = np.eye(18)
    for node, rotation in enumerate(q.reshape(3, 6)[:, 3:]):
        slot = slice(6*node+3, 6*node+6)
        chart[slot, slot] = _exp_terms(rotation)[1]
    for step in (1e-5, 3e-6):
        plus = response(model, element, q+step*direction, origin, 3,
                        stable_translation=True)[0]
        minus = response(model, element, q-step*direction, origin, 3,
                         stable_translation=True)[0]
        derivative = (plus-minus)/(2*step)
        assert np.linalg.norm(derivative-tangent@direction)/max(
            1., np.linalg.norm(tangent@direction)) <= 1e-6
        work = (energy(q+step*direction)-energy(q-step*direction))/(2*step)
        assert abs(work-direction@chart.T@force)/max(1.,abs(work)) <= 1e-6
