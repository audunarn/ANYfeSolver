"""Fixed-fixture F06 development discriminator, not a qualification run.

Coverage uses actual loaded increments relative to virgin physical triads.
No force-pattern replacement, synthetic accepted pose or parameter search is
allowed. Native energy is not inferred from retained resultants: native input
and history preservation, independent shell deformation, dead-load potential
and spatial point-port virtual work are the explicit consequences measured.
"""

from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import b3_ge, stiffened_shell
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from test_stiffened_shell_ge_experimental import _panel


FORCE = np.array((200., 150., 1200.))
LENGTH = 2.
FORCE_SCALE = float(np.linalg.norm(FORCE))
MOMENT_SCALE = FORCE_SCALE*LENGTH


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def _fixture(triangles):
    base = _panel()
    x = np.array((0., .5, 1., 1.5, 2.))
    # One globally quadratic arc is represented exactly by both native Q2
    # elements. Its derivative is continuous at their shared station. Analytic
    # sine tangents do not match the Q2 interpolant of sampled sine positions.
    positions = np.column_stack((x, -.5+.08*x*(2.-x),
                                 .1+.12*x*(2.-x)))
    frames = []
    for point in x:
        axis_x = np.array((1., .16*(1.-point), .24*(1.-point)))
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross((0., 0., 1.), axis_x)
        axis_y /= np.linalg.norm(axis_y)
        frames.append(np.column_stack((axis_x, axis_y, np.cross(axis_x, axis_y))))
    frames = np.array(frames)
    original, _ = base.beams[0].instantiate()
    elastic = .01*original.section.elastic
    section = b3_ge.EllipsoidalGeneralizedSection(elastic, np.eye(6), 1e20, 1e9)
    beams = tuple(b3_ge.define_beam(
        "b3-ge", i+1, (101+2*i, 102+2*i, 103+2*i), positions[2*i:2*i+3],
        frames[2*i:2*i+3], section, np.eye(6), quadrature=original.operator.order)
        for i in range(2))
    shell = replace(base.shell, thickness_m=.007)
    if triangles:
        shell = replace(shell, q4_elements=(),
                        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5)))
    definition = replace(base, shell=shell, beams=beams,
        q4_pullback_policy=("historical_corotational" if triangles else
                            stiffened_shell.Q4_ALGORITHMIC_PULLBACK_V1))
    inputs = dict(shell=asdict(shell), beam_positions=positions, beam_frames=frames,
                  beam_elastic=elastic, beam_metric=np.eye(6), yield_force=1e20,
                  hardening=1e9, inertia=np.eye(6), quadrature=original.operator.order,
                  points=[asdict(item) for item in definition.points],
                  fixed=[asdict(item) for item in definition.fixed],
                  beam_sha256=[item.sha256 for item in beams],
                  q4_pullback_policy=definition.q4_pullback_policy)
    return definition, positions, frames, inputs


def _snapshot(owner, beam_reference, beam_frames0):
    driver, result = owner._finite_driver, owner.recover()
    state = driver._state
    shell_reference = np.asarray(owner._definition.shell.coordinates_m)
    rows = state.q[:owner._shell_count].reshape(-1, 6)
    shell_positions = shell_reference+rows[:, :3]
    shell_frames = Rotation.from_rotvec(rows[:, 3:].copy()).as_matrix()
    if state.shell_anchors is not None:
        shell_frames = shell_frames@state.shell_anchors
    beam_rows = state.q[owner._shell_count:owner._shell_count+driver._beam_nodal].reshape(-1, 6)
    beam_positions = beam_reference+beam_rows[:, :3]
    support = result.support_reactions[:owner._shell_count].reshape(-1, 6)
    support_force = support[:, :3].sum(axis=0)
    # Common global origin; root positions stay equal to their fixed reference.
    support_moment = (support[:, 3:]+np.cross(shell_positions, support[:, :3])).sum(axis=0)
    gaps, virtual_work = [], []
    variation = np.array((.3*LENGTH, -.2*LENGTH, .1*LENGTH, .2, .1, -.3,
                          .1*LENGTH, .4*LENGTH, -.3*LENGTH, -.1, .2, .4))
    for i, (point, joint) in enumerate(zip(owner._definition.points, owner._connections)):
        s, b = owner._shell_index[point.shell_node], owner._beam_index[point.beam_node]
        qs, qb = shell_frames[s], state.frames[b]
        offset0 = beam_reference[b]-shell_reference[s]
        translation = qs.T@(beam_positions[b]-shell_positions[s])-offset0
        orientation = Rotation.from_matrix(beam_frames0[b].T@qs.T@qb).as_rotvec()
        gaps.append(np.r_[translation, orientation])
        # This operator's rows are spatial wrenches; variation contains physical
        # translations and left-spatial rotations, not additive shell shadows.
        response = joint.evaluate(np.array((shell_positions[s], beam_positions[b])),
                                  np.array((qs, qb)), state.multipliers[6*i:6*i+6])
        virtual_work.append(float(response.residual[:12]@variation))
    local = []
    for _, element, slots, _ in owner._shell_parts:
        nodes = slots[::6]//6
        reference, current = shell_reference[nodes], shell_positions[nodes]
        reference_c, current_c = reference-reference.mean(0), current-current.mean(0)
        fit = Rotation.align_vectors(current_c, reference_c)[0]
        displacement = np.empty((len(nodes), 6))
        displacement[:, :3] = fit.inv().apply(current_c)-reference_c
        displacement[:, 3:] = (fit.inv()*Rotation.from_matrix(shell_frames[nodes])).as_rotvec()
        local.append(displacement)
    origin = canonical_json_bytes(_plain((driver._shell_origins, driver._beam_origins)))
    frame_sets = (shell_frames, state.frames, state.cells.reshape(-1, 3, 3))
    fixed_nodes = sorted({owner._shell_index[item.node] for item in owner._definition.fixed
                          if item.domain == "shell"})
    fixed_position_error = float(np.max(np.abs(shell_positions[fixed_nodes]-shell_reference[fixed_nodes])))
    fixed_rotation_error = float(np.max(Rotation.from_matrix(shell_frames[fixed_nodes]).magnitude()))
    gaps = np.array(gaps)
    snapshot = dict(factor=result.load_factor, accepted_factors=result.accepted_factors,
        generation=result.generation, status=result.status, shell_positions=shell_positions,
        shell_frames=shell_frames, beam_positions=beam_positions, beam_frames=state.frames,
        cell_frames=state.cells, retained_resultants=state.resultants, multipliers=state.multipliers,
        support=support, support_force=support_force, support_moment=support_moment,
        gaps=gaps, point_virtual_work=virtual_work,
        fixed_load_linear_potential=-result.load_factor*float(FORCE@(beam_positions[-1]-beam_reference[-1])),
        shell_local_deformation=local, material_origin_sha256=sha256(origin).hexdigest(),
        coordinate_scale=owner._coordinate_scale, constraint_scale=owner._constraint_scale,
        free_equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max,
        independent_gap_max=float(np.max(np.abs(gaps)/np.array((LENGTH, LENGTH, LENGTH, 1., 1., 1.)))),
        force_balance=float(np.linalg.norm(support_force+result.load_factor*FORCE)/FORCE_SCALE),
        moment_balance=float(np.linalg.norm(support_moment+np.cross(beam_positions[-1], result.load_factor*FORCE))/MOMENT_SCALE),
        fixed_position_error=fixed_position_error, fixed_rotation_error=fixed_rotation_error,
        chart_rotation_max=float(np.max(np.linalg.norm(rows[:, 3:], axis=1))),
        local_rotation_max=float(max(np.max(np.linalg.norm(item[:, 3:], axis=1)) for item in local)),
        frame_orthogonality_max=float(max(np.max(np.abs(item@item.transpose(0, 2, 1)-np.eye(3))) for item in frame_sets)),
        minimum_frame_determinant=float(min(np.min(np.linalg.det(item)) for item in frame_sets)))
    return snapshot


def _compare(left, right):
    errors = {}
    for domain in ("shell", "beam"):
        errors[domain+"_position"] = float(np.max(np.abs(left[domain+"_positions"]-right[domain+"_positions"]))/LENGTH)
        relative = left[domain+"_frames"]@right[domain+"_frames"].transpose(0, 2, 1)
        errors[domain+"_orientation"] = float(np.max(Rotation.from_matrix(relative).magnitude()))
    relative = left["cell_frames"].reshape(-1, 3, 3)@right["cell_frames"].reshape(-1, 3, 3).transpose(0, 2, 1)
    errors["cell_orientation"] = float(np.max(Rotation.from_matrix(relative).magnitude()))
    for name, scales in (("support", (FORCE_SCALE,)*3+(MOMENT_SCALE,)*3),
                         ("gaps", (LENGTH,)*3+(1.,)*3),
                         ("retained_resultants", (FORCE_SCALE,)*6+(MOMENT_SCALE,)*12)):
        errors[name] = float(np.max(np.abs(left[name]-right[name])/np.array(scales)))
    errors["multipliers"] = float(np.max(np.abs(left["multipliers"]-right["multipliers"])/
                                       np.tile((FORCE_SCALE,)*3+(MOMENT_SCALE,)*3, len(left["gaps"]))))
    errors["shell_local_deformation"] = float(max(np.max(np.abs(a-b)/
        np.array((LENGTH, LENGTH, LENGTH, 1., 1., 1.))) for a, b in
        zip(left["shell_local_deformation"], right["shell_local_deformation"])))
    errors["point_virtual_work"] = float(np.max(np.abs(np.asarray(left["point_virtual_work"])-right["point_virtual_work"]))/MOMENT_SCALE)
    errors["dead_load_potential"] = abs(left["fixed_load_linear_potential"]-right["fixed_load_linear_potential"])/MOMENT_SCALE
    return errors


def _coverage(history, virgin):
    rotations, commutators = {}, {}
    for domain in ("shell", "beam"):
        frames = [virgin[domain]]+[item[domain+"_frames"] for item in history]
        loaded = [float(np.max(Rotation.from_matrix(frame@frames[0].transpose(0, 2, 1)).magnitude())) for frame in frames[1:]]
        increments = [b@a.transpose(0, 2, 1) for a, b in zip(frames, frames[1:])]
        values = [float(np.max(Rotation.from_matrix(b@a@b.transpose(0, 2, 1)@a.transpose(0, 2, 1)).magnitude()))
                  for a, b in zip(increments, increments[1:])]
        rotations[domain], commutators[domain] = loaded, values
    return dict(loaded_rotation_rad=rotations, actual_commutator_rad=commutators)


@pytest.mark.parametrize("triangles", (False, True), ids=("selected-q4", "all-s3"))
def test_loaded_noncommuting_rebase_consequence(triangles, tmp_path):
    report = dict(stage="fixture", complete=False, force=FORCE.tolist(), factors=(.25, .5, 1.),
                  max_iterations=20, max_backtracks=12, coverage_rotation_floor=.05,
                  coverage_commutator_floor=1e-5, length_scale=LENGTH, force_scale=FORCE_SCALE,
                  moment_work_scale=MOMENT_SCALE, common_origin=(0., 0., 0.), branches={}, comparisons={},
                  energy_limitation="No exposed native total strain energy is inferred from resultants.")
    path = tmp_path/"noncommuting-history.json"
    try:
        definition, beam_reference, frames0, inputs = _fixture(triangles)
        report["fixture"] = inputs
        control = stiffened_shell.create_analysis(definition=definition)
        report["model_identity"] = control._identity
        actions = (stiffened_shell.NodalAction("beam", 105, (*FORCE, 0., 0., 0.)),)
        history = []
        for factor in (.25, .5):
            report["stage"] = f"accept-prefix-{factor}"
            control.solve(actions, (factor,), max_iterations=20, max_backtracks=12)
            history.append(_snapshot(control, beam_reference, frames0))
            report["branches"]["control"] = history.copy()
        prefix = control.checkpoint()
        report["prefix_sha256"] = sha256(prefix).hexdigest()
        report["stage"] = "actual-rebase"
        rebased = stiffened_shell.create_analysis(definition=definition)
        rebased.restore(prefix)
        before = _snapshot(rebased, beam_reference, frames0)
        rebased.rebase_rotations(rebased.recover().generation)
        after = _snapshot(rebased, beam_reference, frames0)
        report["branches"]["rebase"] = [before, after]
        report["comparisons"]["actual_rebase"] = _compare(before, after)
        report["predecessor_invariants"] = dict(
            material_origin=before["material_origin_sha256"] == after["material_origin_sha256"],
            accepted_factors=before["accepted_factors"] == after["accepted_factors"],
            beam_inputs=all(np.array_equal(before[name], after[name]) for name in
                           ("beam_positions", "beam_frames", "cell_frames", "retained_resultants")),
            coordinate_scale=np.array_equal(before["coordinate_scale"], after["coordinate_scale"]),
            constraint_scale=np.array_equal(before["constraint_scale"], after["constraint_scale"]))
        snapshot = rebased.checkpoint()
        report["anchored_sha256"] = sha256(snapshot).hexdigest()
        report["stage"] = "direct-restore"
        resumed = stiffened_shell.create_analysis(definition=definition)
        resumed.restore(snapshot)
        restored = _snapshot(resumed, beam_reference, frames0)
        report["branches"]["resumed"] = [restored]
        report["comparisons"]["direct_restore"] = _compare(after, restored)
        for name, owner in (("control", control), ("resumed", resumed)):
            report["stage"] = "continue-"+name
            owner.solve(actions, (1.,), max_iterations=20, max_backtracks=12)
            report["branches"][name].append(_snapshot(owner, beam_reference, frames0))
        final = report["branches"]["control"][-1]
        continued = report["branches"]["resumed"][-1]
        report["comparisons"]["continued"] = _compare(final, continued)
        virgin = dict(shell=np.tile(np.eye(3), (len(definition.shell.node_ids), 1, 1)), beam=frames0)
        report["virgin_frames"] = virgin
        report["coverage"] = dict(control=_coverage(report["branches"]["control"], virgin),
                                  resumed=_coverage(history[:2]+[continued], virgin))
        report["stage"] = "predicates"
        for values in report["coverage"].values():
            for domain in ("shell", "beam"):
                assert max(values["loaded_rotation_rad"][domain]) >= .05
                assert max(values["actual_commutator_rad"][domain]) >= 1e-5
        assert all(report["predecessor_invariants"].values())
        assert final["material_origin_sha256"] == continued["material_origin_sha256"]
        assert final["accepted_factors"] == continued["accepted_factors"] == (.25, .5, 1.)
        for errors in report["comparisons"].values():
            for name, error in errors.items():
                assert error <= (1e-8 if name in ("point_virtual_work", "dead_load_potential") else 1e-9), (name, error)
        for branch in report["branches"].values():
            for state in branch:
                assert state["status"] == "completed"
                assert state["free_equilibrium_ratio"] <= 1e-8 and state["constraint_max"] <= 1e-9
                assert state["independent_gap_max"] <= 1e-9
                assert state["force_balance"] <= 1e-8 and state["moment_balance"] <= 1e-8
                assert state["fixed_position_error"] <= 1e-12 and state["fixed_rotation_error"] <= 1e-12
                assert state["chart_rotation_max"] < .9*np.pi and state["local_rotation_max"] < .9*np.pi
                assert state["frame_orthogonality_max"] <= 1e-9 and state["minimum_frame_determinant"] > 0.
                assert np.isfinite(state["coordinate_scale"]).all() and np.all(state["coordinate_scale"] > 0.)
                assert np.isfinite(state["constraint_scale"]).all() and np.all(state["constraint_scale"] > 0.)
        report["complete"] = True
    except Exception as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(f"noncommuting-history predicates: {path}")
