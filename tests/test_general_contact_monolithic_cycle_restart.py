"""Deformable-target contact cycle with a checkpoint at the open state."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact.core import suspend_contact
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from scripts.general_contact_d1_free_target_press_probe import (
    build_free_target_case,
)


def _case():
    return build_free_target_case(
        4, steps_per_segment=2, four_phase=True,
        enforcement_solver="monolithic_active_set",
    )


def _solve(*, stop=None, checkpoint=None, snapshots=False):
    model, definition, settings, program = _case()
    kwargs = {"max_load_factor": stop} if stop is not None else {}
    return solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True, restart_checkpoint=checkpoint,
        record_increment_snapshots=snapshots,
        **kwargs,
    )


def _check_target_clamp_wrench(model, manager, snapshot, record, support):
    certificate = record["contact"]
    assert certificate["source_pressure_node_ids"] == list(
        manager.law.pressure_mode_ids)
    support_area = np.asarray(certificate["reference_support_m2"])
    pressure = np.asarray(certificate["numerical_pressure_pa"])
    contact = linearize(manager, snapshot.displacements,
                        pressure*support_area, tangent=False)
    np.testing.assert_allclose(contact.state.pressure, pressure,
                               rtol=1e-9, atol=1e-12)
    with suspend_contact():
        internal, _K, _states = _assemble_nonlinear_system(
            model, snapshot.displacements, snapshot.element_states, 5,
            tangent=False, kinematics="von_karman",
            require_full_coordinates=True,
        )
    residual = internal + contact.force
    clamp_force = np.zeros(3)
    clamp_moment = np.zeros(3)
    clamp_components = np.zeros(6)
    for node in (manager.law.target_node_ids[0],
                 manager.law.target_node_ids[-1]):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        reaction = residual[dofs]
        reference = model.mesh.nodes[node]
        current = (np.array((reference.x, reference.y, reference.z))
                   + snapshot.displacements[dofs[:3]])
        clamp_components += reaction
        clamp_force += reaction[:3]
        clamp_moment += np.cross(current, reaction[:3]) + reaction[3:6]
    np.testing.assert_allclose(clamp_components, support["target_left_clamp"],
                               rtol=0.0, atol=1e-12)
    target_force = np.zeros(3)
    target_moment = np.zeros(3)
    for node in manager.law.target_node_ids:
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        reference = model.mesh.nodes[node]
        current = (np.array((reference.x, reference.y, reference.z))
                   + snapshot.displacements[dofs[:3]])
        physical = -contact.force[dofs]
        target_force += physical[:3]
        target_moment += np.cross(current, physical[:3]) + physical[3:6]
    force_scale = max(np.linalg.norm(target_force), 1e-8)
    moment_scale = max(np.linalg.norm(target_moment),
                       1.3*np.linalg.norm(target_force), 1e-8)
    assert np.linalg.norm(clamp_force + target_force)/force_scale < 1e-6
    # Tolerance drift under the adopted S3 VK law: VK membrane forces are not
    # rigid-rotation objective, so the target's internal forces carry a net
    # moment (measured max 1.467e-6 of moment_scale over the checked
    # snapshots; 4.2e-7 before VK at the shared press pose, where the
    # free-node residual moment is 8e-16). Tolerance = max(1e-6, 1.5 x
    # measured) = 2.2e-6.
    assert np.linalg.norm(clamp_moment + target_moment)/moment_scale < 2.2e-6


def _check_smooth_path_work(model, manager, snapshot, record):
    """Check both-body mapped work only while the local overlap branch persists."""
    certificate = record["contact"]
    nodal_forces = (np.asarray(certificate["numerical_pressure_pa"])
                    * np.asarray(certificate["reference_support_m2"]))
    pose = snapshot.displacements
    base = linearize(manager, pose, nodal_forces, tangent=True)
    assert np.count_nonzero(nodal_forces > 0.0) > 0
    columns = ((3, 0), (3, 2), (13, 0), (13, 2), (13, 4))
    directions = []
    for node, axis in columns:
        direction = np.zeros_like(pose)
        direction[model.mesh.dof_manager.get_node_dofs(node)[axis]] = 1.0
        directions.append(direction)
    for weights in ((0.4, -0.3, 0.6, -0.5, 0.2),
                    (-0.2, 0.5, 0.3, 0.4, -0.6)):
        directions.append(sum((weight*direction for weight, direction
                               in zip(weights, directions[:len(columns)], strict=True)),
                              np.zeros_like(pose)))
    for axis in (2, 3, 4):
        direction = np.zeros_like(pose)
        direction[model.mesh.dof_manager.get_node_dofs(14)[axis]] = 1.0
        directions.append(direction)
    for direction in directions:
        step = 1e-7
        plus = linearize(manager, pose+step*direction, nodal_forces,
                         tangent=False)
        minus = linearize(manager, pose-step*direction, nodal_forces,
                          tangent=False)
        assert (base.state.overlap_signature == plus.state.overlap_signature
                == minus.state.overlap_signature)
        assert (base.state.integration_triangles
                == plus.state.integration_triangles
                == minus.state.integration_triangles)
        observed = (plus.force-minus.force)/(2*step)
        predicted = base.tangent @ direction
        scale = max(np.linalg.norm(observed), np.linalg.norm(predicted))
        assert scale > 1e-7
        assert np.linalg.norm(observed-predicted)/scale < 1e-5
        work = (np.dot(nodal_forces, plus.state.normalized_gap
                       -minus.state.normalized_gap)/(2*step))
        assert abs(work+float(base.force @ direction)) < 1e-12


def _independent_skin_vertices(model, element, displacement, side):
    """Small-rotation physical skin built without the contact work adapter."""
    nodes = [model.mesh.nodes[node_id] for node_id in element.node_ids]
    reference = np.asarray([(node.x, node.y, node.z) for node in nodes])
    if len(nodes) == 4:
        director = np.cross(reference[1]-reference[0],
                            reference[3]-reference[0])
        director /= np.linalg.norm(director)
        offset = (-0.5 if side == "bottom" else 0.5)*element.thickness
    else:
        director = (np.asarray(element.reference_normal)
                    * element.director_polarity)
        offset = (0.5*element.thickness
                  - element.reference_surface_offset)
    vertices = []
    for node_id, point in zip(element.node_ids, reference, strict=True):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        vertices.append(point + displacement[dofs[:3]]
                        + offset*(director
                                  + np.cross(displacement[dofs[3:6]],
                                             director)))
    return np.asarray(vertices)


def test_physical_skin_path_matches_linear_swept_certificate_assumption():
    """The admitted small-rotation work map is affine along a solver trial."""
    model, definition, settings, _program = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    rng = np.random.default_rng(20260929)
    start = rng.uniform(-0.002, 0.002, manager.n_dof)
    end = start + rng.uniform(-0.01, 0.01, manager.n_dof)
    first = manager._positions(start)
    last = manager._positions(end)
    for fraction in (0.0, 0.17, 0.5, 0.83, 1.0):
        displacement = (1-fraction)*start + fraction*end
        mapped = manager._positions(displacement)
        for side_index, side_name in enumerate(("bottom", "top")):
            by_node = {}
            for element_id in definition.surfaces[side_index].element_ids:
                element = model.mesh.elements[element_id]
                vertices = _independent_skin_vertices(
                    model, element, displacement, side_name,
                )
                by_node.update(zip(element.node_ids, vertices, strict=True))
            node_ids = (manager.law.source_node_ids if side_index == 0
                        else manager.law.target_node_ids)
            independent = np.asarray([by_node[node_id] for node_id in node_ids])
            np.testing.assert_allclose(mapped[side_index], independent,
                                       rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(
                mapped[side_index],
                (1-fraction)*first[side_index]+fraction*last[side_index],
                rtol=0.0, atol=1e-14,
            )


def _sampled_physical_gap(model, definition, displacement, subdivisions):
    """Independent vertical projection onto each current S3 triangle."""
    sources = [_independent_skin_vertices(model, model.mesh.elements[element_id],
                                         displacement, "bottom")
               for element_id in definition.surfaces[0].element_ids]
    targets = [_independent_skin_vertices(model, model.mesh.elements[element_id],
                                         displacement, "top")
               for element_id in definition.surfaces[1].element_ids]
    minimum = np.inf
    included = 0
    for source in sources:
        for a in np.linspace(0.0, 1.0, subdivisions+1):
            for b in np.linspace(0.0, 1.0, subdivisions+1):
                point = ((1-a)*(1-b)*source[0] + a*(1-b)*source[1]
                         + a*b*source[2] + (1-a)*b*source[3])
                for target in targets:
                    xy = np.column_stack((target[1, :2]-target[0, :2],
                                          target[2, :2]-target[0, :2]))
                    bary = np.linalg.solve(xy, point[:2]-target[0, :2])
                    if (np.min(bary) < -1e-12
                            or np.sum(bary) > 1.0+1e-12):
                        continue
                    projected = target[0] + bary[0]*(target[1]-target[0]) \
                        + bary[1]*(target[2]-target[0])
                    normal = np.cross(target[1]-target[0],
                                      target[2]-target[0])
                    normal /= np.linalg.norm(normal)
                    minimum = min(minimum, float((point-projected) @ normal))
                    included += 1
    assert included > 0
    return minimum


def _bounded_projected_gap(model, definition, displacement, gap_budget):
    """Bound target-normal skin gap on the vertical XY-overlap footprint.

    Q4 position, target-triangle barycentric coordinates and signed distance
    are bilinear in each Q4 parameter rectangle. Their extrema therefore
    occur at its four corners. Partial-overlap boxes are subdivided rather
    than treating an outside corner as a physical contact point. The bound
    is exact in arithmetic for this planar fixture; explicit conditioning
    and roundoff guards limit its finite-precision use.
    """
    sources = [_independent_skin_vertices(model, model.mesh.elements[element_id],
                                         displacement, "bottom")
               for element_id in definition.surfaces[0].element_ids]
    targets = [_independent_skin_vertices(model, model.mesh.elements[element_id],
                                         displacement, "top")
               for element_id in definition.surfaces[1].element_ids]
    lower = np.inf
    retained = 0
    visited = 0
    scale = max(1.0, *(float(np.max(np.abs(face)))
                       for face in (*sources, *targets)))
    for source in sources:
        for target in targets:
            xy = np.column_stack((target[1, :2]-target[0, :2],
                                  target[2, :2]-target[0, :2]))
            inverse = np.linalg.inv(xy)
            condition = (np.linalg.norm(xy, ord=np.inf)
                         * np.linalg.norm(inverse, ord=np.inf))
            assert np.isfinite(condition) and condition <= 1e6, (
                "independent gap XY projection is ill-conditioned"
            )
            roundoff = 256*np.finfo(float).eps*condition*scale
            bary_roundoff = (128*np.finfo(float).eps
                             * (1.0+np.linalg.norm(inverse, ord=np.inf)*scale))
            normal = np.cross(target[1]-target[0], target[2]-target[0])
            normal /= np.linalg.norm(normal)
            assert normal[2] > 0.5
            boxes = [(0.0, 1.0, 0.0, 1.0, 0)]
            while boxes:
                a0, a1, b0, b1, depth = boxes.pop()
                visited += 1
                assert visited <= 100_000, "independent gap box budget exhausted"
                parameters = ((a0, b0), (a1, b0), (a1, b1), (a0, b1))
                corners = np.asarray([
                    (1-a)*(1-b)*source[0] + a*(1-b)*source[1]
                    + a*b*source[2] + (1-a)*b*source[3]
                    for a, b in parameters
                ])
                transverse = (corners[:, :2]-target[0, :2]) @ inverse.T
                barycentric = np.column_stack((
                    1.0-transverse.sum(axis=1), transverse,
                ))
                if np.any(np.max(barycentric, axis=0) < -bary_roundoff):
                    continue
                corner_gap = (corners-target[0]) @ normal
                local_lower = float(np.min(corner_gap)-roundoff)
                interior_corners = np.all(barycentric >= bary_roundoff, axis=1)
                if np.any(interior_corners & (corner_gap < -gap_budget-roundoff)):
                    raise AssertionError(
                        "independent projected gap violates budget at interior corner"
                    )
                if local_lower >= -gap_budget:
                    lower = min(lower, local_lower)
                    retained += 1
                    continue
                if np.all(np.min(barycentric, axis=0) >= bary_roundoff):
                    raise AssertionError(
                        f"independent projected gap violates budget: {local_lower} m"
                    )
                assert depth < 14, "independent gap bound unresolved at overlap edge"
                amid = 0.5*(a0+a1)
                bmid = 0.5*(b0+b1)
                boxes.extend((
                    (a0, amid, b0, bmid, depth+1),
                    (amid, a1, b0, bmid, depth+1),
                    (a0, amid, bmid, b1, depth+1),
                    (amid, a1, bmid, b1, depth+1),
                ))
    assert retained > 0
    return lower


def _check_independent_skin_vertices(model, definition, manager, displacement):
    mapped_source, mapped_target = manager._positions(displacement)
    for surface, side, node_ids, mapped in (
        (definition.surfaces[0], "bottom", manager.law.source_node_ids,
         mapped_source),
        (definition.surfaces[1], "top", manager.law.target_node_ids,
         mapped_target),
    ):
        by_node = {}
        for element_id in surface.element_ids:
            element = model.mesh.elements[element_id]
            vertices = _independent_skin_vertices(
                model, element, displacement, side,
            )
            by_node.update(zip(element.node_ids, vertices, strict=True))
        np.testing.assert_allclose(
            np.asarray([by_node[node_id] for node_id in node_ids]), mapped,
            rtol=0.0, atol=1e-12,
        )


def test_free_target_press_slide_open_recontact_split_restart():
    full = _solve(snapshots=True)
    assert full.status == "completed"
    assert len(full.steps) == 8
    model, definition, settings, program = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    history = full.info["force_displacement_history"]
    assert [snapshot.step_index for snapshot in full.snapshots] == list(range(1, 9))
    assert history[1]["contact"]["integration_force"][2] > 0.0
    assert history[5]["contact"]["active_pressure_dofs"] == 0
    assert history[5]["contact"]["minimum_gap"] > 0.0
    assert history[6]["contact"]["integration_force"][2] > 0.0
    for row in history:
        source = np.asarray(row["contact"]["integration_force"])
        target = np.asarray(row["contact"]["target_force"])
        np.testing.assert_allclose(source+target, 0.0, atol=1e-9)
        source_moment = np.asarray(row["contact"]["integration_moment"])
        target_moment = np.asarray(row["contact"]["target_moment"])
        np.testing.assert_allclose(source_moment+target_moment, 0.0, atol=1e-9)
        assert row["contact"]["gap_tolerance_met"]
        assert row["complementarity_relative_force"] <= 1e-8
        step = int(row["step_index"])
        support = full.steps[step-1].support_reactions
        assert abs(support["target_left_clamp"][2]-source[2]) <= 3e-7
        external_z = float(np.sum(program.force_values_at(step)))
        assert abs(sum(values[2] for values in support.values())
                   + external_z) <= 3e-7
        _check_target_clamp_wrench(model, manager, full.snapshots[step-1],
                                   row, support)
        if step in (4, 8):
            _check_smooth_path_work(model, manager, full.snapshots[step-1], row)
        pose = full.snapshots[step-1].displacements
        _check_independent_skin_vertices(model, definition, manager, pose)
        sampled_16 = _sampled_physical_gap(model, definition, pose, 16)
        sampled_32 = _sampled_physical_gap(model, definition, pose, 32)
        gap_budget = definition.interactions[0].gap_tolerance
        bounded_gap = _bounded_projected_gap(
            model, definition, pose, gap_budget,
        )
        assert bounded_gap >= -gap_budget
        assert bounded_gap <= sampled_32+1e-12
        assert sampled_32 >= -gap_budget
        assert abs(sampled_32-sampled_16) <= 0.1*gap_budget

    # A physical vertex deliberately pushed through the target must fail the
    # independent bound, so an empty or misoriented overlap cannot pass it.
    penetrated = full.snapshots[1].displacements.copy()
    penetrated[model.mesh.dof_manager.get_node_dofs(2)[2]] -= 1e-3
    with pytest.raises(AssertionError, match="violates budget"):
        _bounded_projected_gap(model, definition, penetrated,
                               definition.interactions[0].gap_tolerance)

    opened = _solve(stop=0.75, snapshots=True)
    assert opened.status == "completed"
    assert len(opened.steps) == 6
    assert [snapshot.step_index for snapshot in opened.snapshots] == list(range(1, 7))
    assert opened.info["force_displacement_history"][-1]["contact"][
        "active_pressure_dofs"] == 0
    checkpoint = opened.to_restart_checkpoint()
    changed_model, changed_definition, changed_settings, changed_program = _case()
    interaction = changed_definition.interactions[0]
    changed_definition = replace(changed_definition, interactions=(
        replace(interaction, normal_stiffness=2*interaction.normal_stiffness),
    ))
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            changed_model, contact=changed_definition,
            contact_settings=changed_settings,
            contact_path_program=changed_program, max_iterations=18,
            restart_checkpoint=checkpoint,
        )
    resumed = _solve(checkpoint=checkpoint, snapshots=True)
    assert resumed.status == "completed"
    assert [snapshot.step_index for snapshot in resumed.snapshots] == [7, 8]
    np.testing.assert_array_equal(resumed.snapshots[-1].displacements,
                                  full.snapshots[-1].displacements)
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["integration_force"],
        full.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1e-10,
    )
