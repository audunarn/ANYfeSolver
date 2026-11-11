"""Physical virtual-work controls for an internal fixed S3 friction pair."""

import numpy as np
import pytest

from anyfesolver import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.friction_point import (
    FrictionPointState, initial_friction_point,
    objective_relative_tangent_increment,
)
from anyfesolver.general_contact.s3_friction_pair import S3FrictionPointPair
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint


def _case(*, maximum_active_columns=72):
    model = FEModel("s3_fixed_material_friction_pair")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    positions = {
        1: (0.4, 0.2, 0.065), 2: (0.9, 0.2, 0.065),
        3: (0.4, 0.7, 0.065), 4: (0.0, 0.0, 0.0),
        5: (1.0, 0.0, 0.0), 6: (0.0, 1.0, 0.0),
    }
    for node, position in positions.items():
        model.add_node(node, *position)
    for element, nodes in ((1, (1, 2, 3)), (2, (4, 5, 6))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    source = S3SkinMaterialPoint.from_model(
        model, 1, 1, "bottom", (0.3, 0.3, 0.4),
    )
    target = S3SkinMaterialPoint.from_model(
        model, 2, 2, "top", (0.05, 0.55, 0.4),
    )
    pair = S3FrictionPointPair(
        source, target, reference_weight_m2=0.01,
        friction_coefficient=0.3, tangential_stiffness_Pa_per_m=1.0e9,
        maximum_active_columns=maximum_active_columns,
    )
    committed = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3):
        committed[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.015
    history = initial_friction_point(target.physical_pose(committed).material_frame)
    return model, positions, pair, committed, history


def _translation_mode(model, nodes, component):
    mode = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in nodes:
        mode[model.mesh.dof_manager.get_node_dofs(node)[component]] = 1.0
    return mode


def test_retained_nonproportional_slip_has_no_rigid_spin_work_or_commit_jump():
    model, positions, pair, origin, history = _case()
    old = origin + .0001*_translation_mode(model,(1,2,3),0)
    retained = pair.evaluate(origin,old,history,normal_pressure_Pa=2e6).trial.candidate_state
    u = old + .0002*_translation_mode(model,(1,2,3),1)
    trial = pair.evaluate(old,u,retained,normal_pressure_Pa=2e6)
    assert trial.trial.branch == "stick"
    spin = np.zeros(u.size)
    axis = np.array((0.,0.,1.))
    for node, reference in positions.items():
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        spin[dofs[:3]] = np.cross(axis,np.asarray(reference)+u[dofs[:3]])
        spin[dofs[3:6]] = axis
    assert abs(float(trial.structural_force@spin)) <= 1e-8
    refreshed = pair.evaluate(u,u,trial.trial.candidate_state,normal_pressure_Pa=2e6)
    np.testing.assert_allclose(refreshed.structural_force,trial.structural_force,rtol=1e-10,atol=1e-8)


def test_s3_pair_force_is_objective_virtual_work_and_balanced():
    model, positions, pair, committed, history = _case()
    candidate = committed.copy()
    for node in (1, 2, 3):
        candidate[model.mesh.dof_manager.get_node_dofs(node)[0]] += 0.0002
    result = pair.evaluate(committed, candidate, history,
                           normal_pressure_Pa=2.0e6)
    assert result.trial.branch == "stick"
    np.testing.assert_allclose(result.relative_increment,
                               (0.0002, 0.0, 0.0), atol=1e-15)
    np.testing.assert_allclose(result.trial.traction,
                               (-0.2e6, 0.0, 0.0), atol=1e-7)
    source_x = _translation_mode(model, (1, 2, 3), 0)
    target_x = _translation_mode(model, (4, 5, 6), 0)
    assert float(result.structural_force @ source_x) == pytest.approx(-2000.0)
    assert float(result.structural_force @ target_x) == pytest.approx(2000.0)
    for component in range(3):
        common = _translation_mode(model, range(1, 7), component)
        assert abs(float(result.structural_force @ common)) <= 1.0e-8

    direction = np.linspace(-0.03, 0.04, candidate.size)
    old_source = pair.source.physical_pose(committed)
    old_target = pair.target.physical_pose(committed)

    def slip(u):
        new_source = pair.source.physical_pose(u)
        new_target = pair.target.physical_pose(u)
        return np.asarray(objective_relative_tangent_increment(
            source_committed=old_source.position,
            target_committed=old_target.position,
            source_candidate=new_source.position,
            target_candidate=new_target.position,
            target_frame_committed=old_target.material_frame,
            target_frame_candidate=new_target.material_frame,
        ))

    step = 3.0e-6
    finite_work = 0.01 * float(np.asarray(result.trial.traction) @ (
        (slip(candidate + step*direction)
         - slip(candidate - step*direction)) / (2.0*step)
    ))
    assert float(result.structural_force @ direction) == pytest.approx(
        finite_work, rel=1.0e-7, abs=1.0e-7,
    )
    # A common infinitesimal rigid rotation must do no frictional work.
    rotation = np.array((0.0, 0.0, 1.0))
    common_rotation = np.zeros_like(candidate)
    for node, coordinates in positions.items():
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        current_node = np.asarray(coordinates) + candidate[dofs[:3]]
        common_rotation[dofs[:3]] = np.cross(rotation, current_node)
        common_rotation[dofs[3:6]] = rotation
    assert abs(float(result.structural_force @ common_rotation)) <= 1.0e-7
    assert abs(result.structural_force[model.mesh.dof_manager.get_node_dofs(1)[4]]) > 0.0
    drill = np.zeros_like(candidate)
    drill[model.mesh.dof_manager.get_node_dofs(5)[5]] = 1.0
    assert float(result.structural_force @ drill) == 0.0


def test_s3_pair_trials_do_not_commit_and_opening_releases_force():
    model, _, pair, committed, history = _case()
    candidate = committed.copy()
    for node in (1, 2, 3):
        candidate[model.mesh.dof_manager.get_node_dofs(node)[0]] += 0.0002
    first = pair.evaluate(committed, candidate, history,
                          normal_pressure_Pa=2.0e6)
    second = pair.evaluate(committed, candidate, history,
                           normal_pressure_Pa=2.0e6)
    np.testing.assert_array_equal(first.structural_force, second.structural_force)
    assert first.trial.candidate_state == second.trial.candidate_state
    assert history.partner_key is None
    assert history.elastic_slip == (0.0, 0.0, 0.0)
    open_state = pair.evaluate(committed, candidate, history,
                               normal_pressure_Pa=0.0)
    assert open_state.trial.branch == "open"
    np.testing.assert_array_equal(open_state.structural_force,
                                  np.zeros_like(open_state.structural_force))
    assert open_state.trial.candidate_state.partner_key is None
    separated_after_stick = pair.evaluate(
        candidate, candidate, first.trial.candidate_state,
        normal_pressure_Pa=0.0,
    )
    assert separated_after_stick.trial.branch == "open"
    assert separated_after_stick.trial.released_storage_density == pytest.approx(20.0)
    np.testing.assert_array_equal(separated_after_stick.structural_force,
                                  np.zeros_like(separated_after_stick.structural_force))

    wrong = FrictionPointState(
        history.material_frame, history.elastic_slip, partner_key="wrong-pair",
    )
    with pytest.raises(ValueError, match="different material pair"):
        pair.evaluate(committed, candidate, wrong, normal_pressure_Pa=2.0e6)
    with pytest.raises(MemoryError, match="active-column budget"):
        limited = S3FrictionPointPair(
            pair.source, pair.target, 0.01, 0.3, 1.0e9,
            maximum_active_columns=1,
        )
        limited.evaluate(committed, candidate, history,
                         normal_pressure_Pa=2.0e6)


@pytest.mark.parametrize("slip,branch", [
    (0.0002, "stick"), (0.001, "slip"),
])
def test_s3_pair_bounded_tangent_matches_independent_direction(slip, branch):
    model, _, pair, committed, history = _case()
    candidate = committed.copy()
    for node in (1, 2, 3):
        candidate[model.mesh.dof_manager.get_node_dofs(node)[0]] += slip
    tangent = pair.numerical_tangent(
        committed, candidate, history,
        normal_pressure_Pa=2.0e6,
        steps_by_dof=np.full(candidate.size, 1.0e-7),
        maximum_candidate_evaluations=100,
    )
    assert tangent.evaluation.trial.branch == branch
    assert tangent.candidate_evaluations == 1 + 2*len(
        tangent.evaluation.active_columns,
    )
    direction = np.linspace(-0.03, 0.04, candidate.size)
    step = 2.0e-5
    plus = pair.evaluate(committed, candidate + step*direction, history,
                         normal_pressure_Pa=2.0e6)
    minus = pair.evaluate(committed, candidate - step*direction, history,
                          normal_pressure_Pa=2.0e6)
    assert plus.trial.branch == minus.trial.branch == branch
    finite = (plus.structural_force - minus.structural_force)/(2.0*step)
    error = tangent.tangent @ direction - finite
    for local_dofs in (slice(0, 3), slice(3, 6)):
        indices = np.asarray([
            model.mesh.dof_manager.get_node_dofs(node)[local_dofs]
            for node in range(1, 7)
        ]).reshape(-1)
        assert (np.linalg.norm(error[indices]) / np.linalg.norm(finite[indices])
                <= 1.0e-5)
    if branch == "stick":
        defect = np.linalg.norm(tangent.tangent - tangent.tangent.T)
        assert defect / np.linalg.norm(tangent.tangent) <= 1.0e-5
    assert history.partner_key is None


def test_s3_pair_tangent_open_budget_and_branch_kink():
    model, _, pair, committed, history = _case()
    steps = np.full(committed.size, 1.0e-7)
    opened = pair.numerical_tangent(
        committed, committed, history, normal_pressure_Pa=0.0,
        steps_by_dof=steps, maximum_candidate_evaluations=1,
    )
    assert opened.evaluation.trial.branch == "open"
    assert opened.candidate_evaluations == 1
    np.testing.assert_array_equal(opened.tangent,
                                  np.zeros_like(opened.tangent))

    moving = committed.copy()
    for node in (1, 2, 3):
        moving[model.mesh.dof_manager.get_node_dofs(node)[0]] += 0.0002
    with pytest.raises(MemoryError, match="evaluation budget"):
        pair.numerical_tangent(
            committed, moving, history, normal_pressure_Pa=2.0e6,
            steps_by_dof=steps, maximum_candidate_evaluations=1,
        )
    at_yield = committed.copy()
    for node in (1, 2, 3):
        at_yield[model.mesh.dof_manager.get_node_dofs(node)[0]] += 0.0006
    with pytest.raises(ValueError, match="stick/slip branch"):
        pair.numerical_tangent(
            committed, at_yield, history, normal_pressure_Pa=2.0e6,
            steps_by_dof=steps, maximum_candidate_evaluations=100,
        )
