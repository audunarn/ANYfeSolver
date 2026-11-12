"""Supporting static development tests with independent line/cap potentials.

Direct Lagrange polynomials and SciPy quadrature define the line reference;
the physical-circle oracle supplies cap gaps. No manager integration, assembly,
stationary-point equation, or beam stiffness implementation is used by them.
"""

from dataclasses import FrozenInstanceError, replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq, root

from anyfesolver import (BoundaryCondition, CancellationToken, DisplacementControl,
                      LoadCase, RigidContactBody, SolveCancelled,
                      report_general_contact_capability, solve_static_nonlinear,
                      validate_general_contact)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.general_contact.beam_hemisphere_cylinder import (
    HemisphereCylinderManager, PROFILE,
)
from anyfesolver.general_contact.beam_radial_ends import PROFILE as LEGACY_PROFILE
from test_general_contact_hemisphere_cylinder_work import _reference
from test_general_contact_radial_beam_ends import _case, _static


def _support_case(quadratic=False, ke=1000.0):
    model, definition, settings = _case("cylinder", quadratic, ke)
    return model, definition, replace(settings, beam_end_cap_work="hemisphere_support_dev")


def _shapes(nodes):
    s = Polynomial([0.0, 1.0])
    return [1 - s, s] if nodes == 2 else [(1 - s) * (1 - 2 * s), 4 * s * (1 - s), s * (2 * s - 1)]


def _line_reference(positions, radius, point, axis, stiffness, length, *, forces=True):
    shapes = _shapes(len(positions))
    axis = np.asarray(axis) / np.linalg.norm(axis)
    relative = positions - point
    projected = relative - np.outer(relative @ axis, axis)
    lanes = [sum((shapes[i] * projected[i, j] for i in range(len(positions))), Polynomial([0.0]))
             for j in range(3)]
    distance_squared = sum((lane * lane for lane in lanes), Polynomial([0.0]))
    roots = (distance_squared - radius**2).roots()
    cuts = [0.0] + sorted(float(r.real) for r in roots
                         if abs(r.imag) < 1e-9 and 0 < r.real < 1) + [1.0]

    def radial_work(s):
        radial = np.array([lane(s) for lane in lanes])
        distance = np.linalg.norm(radial)
        return min(distance - radius, 0.0), radial / distance

    def integrate(function):
        return sum(quad(function, a, b, epsabs=2e-15, epsrel=2e-10, limit=100)[0]
                   for a, b in zip(cuts, cuts[1:]))

    energy = integrate(lambda s: 0.5 * stiffness * radial_work(s)[0]**2 * length)
    force = np.zeros_like(positions)
    if not forces:
        return energy, force
    for i, shape in enumerate(shapes):
        for j in range(3):
            force[i, j] = integrate(lambda s: stiffness * radial_work(s)[0]
                                    * radial_work(s)[1][j] * shape(s) * length)
    return energy, force


def _positions(model, displacement):
    return np.array([node.coords() + displacement[model.mesh.dof_manager.get_node_dofs(i)[:3]]
                     for i, node in sorted(model.mesh.nodes.items())])


def _potential(model, definition, displacement):
    positions = _positions(model, displacement)
    surface, body, law = definition.surfaces[0], definition.rigid_bodies[0], definition.interactions[0]
    reference = np.array([node.coords() for _, node in sorted(model.mesh.nodes.items())])
    length = np.linalg.norm(reference[-1] - reference[0])
    line, _ = _line_reference(positions, surface.beam_radius + body.radius,
                              np.asarray(body.point), np.asarray(body.normal), law.line_stiffness,
                              length, forces=False)
    cap = sum(0.5 * law.endpoint_stiffness * min(_reference(positions, end, sign,
              surface.beam_radius, np.asarray(body.point), np.asarray(body.normal), body.radius)[0], 0.0)**2
              for end, sign in ((0, -1), (len(positions) - 1, 1)))
    return line, cap


@pytest.mark.parametrize("trial", ["b2_sphere", "b3_sphere", "b3_rim"])
def test_manager_complete_work_from_independent_line_and_cap_energy(trial):
    quadratic = trial != "b2_sphere"
    model, definition, settings = _support_case(quadratic)
    manager = HemisphereCylinderManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    if trial == "b3_rim":
        u[model.mesh.dof_manager.get_node_dofs(1)[0]] = -0.0035
        u[model.mesh.dof_manager.get_node_dofs(2)[:2]] = (0.1587 - 0.212, 0.004)
    else:
        u[model.mesh.dof_manager.get_node_dofs(1)[:3]] = (-0.004, 0.0003, 0.0002)
        if quadratic:
            u[model.mesh.dof_manager.get_node_dofs(2)[:3]] = (-0.001, 0.002, 0.0)
    residual, tangent, result = manager.evaluate(u)
    assert result.profile == PROFILE and result.active_endpoints == 1
    assert result.endpoint_fields[0]["branch"] == ("rim" if trial == "b3_rim" else "sphere")
    assert {field["node"] for field in result.endpoint_fields} == {1, 3 if quadratic else 2}
    line, cap = _potential(model, definition, u)
    assert line > 0 and cap > 0
    assert_allclose(result.line_contact_energy, line, rtol=2e-7, atol=2e-13)
    assert_allclose(result.endpoint_contact_energy, cap, rtol=2e-10, atol=2e-13)
    assert_allclose(np.asarray(result.beam_force) + result.rigid_force, 0, atol=2e-12)
    assert_allclose(np.asarray(result.beam_moment) + result.rigid_moment, 0, atol=2e-12)
    translations = np.concatenate([model.mesh.dof_manager.get_node_dofs(i)[:3]
                                   for i in range(1, 4 if quadratic else 3)])
    rotations = np.setdiff1d(np.arange(len(u)), translations)
    assert_allclose(residual[rotations], 0, atol=0)
    assert tangent[rotations, :].nnz == 0 and tangent[:, rotations].nnz == 0
    assert_allclose(tangent.toarray(), tangent.toarray().T, atol=2e-10, rtol=0)
    if trial == "b3_rim":
        doubled = replace(definition, interactions=(replace(definition.interactions[0],
                                                           endpoint_stiffness=2000.0),))
        doubled_force, doubled_tangent, doubled_result = HemisphereCylinderManager(
            model, doubled, settings).evaluate(u)
        middle_dofs = model.mesh.dof_manager.get_node_dofs(2)[:3]
        endpoint_dofs = model.mesh.dof_manager.get_node_dofs(1)[:3]
        assert np.linalg.norm((doubled_force - residual)[middle_dofs]) > 1e-4
        assert np.linalg.norm((doubled_tangent - tangent)[middle_dofs, :][:, endpoint_dofs].toarray()) > 1e-3
        assert doubled_result.line_contact_energy == result.line_contact_energy
        assert_allclose(doubled_result.endpoint_contact_energy, 2 * result.endpoint_contact_energy,
                        atol=2e-13, rtol=2e-12)
    reference_gradient = np.zeros_like(u)
    h = 3e-7
    for dof in translations:
        delta = np.zeros_like(u)
        delta[dof] = h
        reference_gradient[dof] = (sum(_potential(model, definition, u + delta))
                                   - sum(_potential(model, definition, u - delta))) / (2 * h)
    assert_allclose(residual, reference_gradient, rtol=1e-5, atol=1e-7)
    direction = np.zeros_like(u)
    direction[translations] = np.random.default_rng(748).normal(size=len(translations))
    direction /= np.linalg.norm(direction)
    expected = tangent @ direction
    errors = []
    reference_errors = []
    for h in (3e-7, 1e-7):
        plus = manager.evaluate(u + h * direction, tangent=False)[0]
        minus = manager.evaluate(u - h * direction, tangent=False)[0]
        errors.append(np.linalg.norm((plus - minus) / (2 * h) - expected)
                      / max(np.linalg.norm(expected), 1e-12))
        ep, em, e0 = (sum(_potential(model, definition, v))
                      for v in (u + h * direction, u - h * direction, u))
        assert_allclose((ep - em) / (2 * h), residual @ direction, rtol=1e-5, atol=1e-7)
        second = (ep - 2 * e0 + em) / h**2
        reference_errors.append(abs(second - direction @ expected)
                                / max(abs(direction @ expected), 1.0))
    assert min(errors) < 1e-5, errors
    assert min(reference_errors) < 1e-5, reference_errors


def _bar_case(quadratic=False):
    if not quadratic:
        model, definition, settings, load = _static("cylinder")
    else:
        model, definition, settings = _case("cylinder", True, 1e5)
        definition = replace(definition, interactions=(replace(definition.interactions[0], gap_tolerance=1e-5),))
        model.add_boundary_condition(BoundaryCondition("root", (3,),
            {d: 0.0 for d in ("ux", "uy", "uz", "rx", "ry", "rz")}))
        model.add_boundary_condition(BoundaryCondition("guide", (1, 2),
            {d: 0.0 for d in ("uy", "uz", "rx", "ry", "rz")}))
        load = LoadCase("press")
        load.add_nodal_load(1, forces=(-0.2, 0.0, 0.0))
    return model, definition, replace(settings, beam_end_cap_work="hemisphere_support_dev"), load


def _bar_reference(quadratic, prescribed=None):
    """Independent axial FE equilibrium, with contact forces integrated directly."""
    nodes = 3 if quadratic else 2
    reference = np.array([0.112, 0.212, 0.312] if quadratic else [0.112, 0.312])
    length, EA, k, ke = 0.2, 1.0, 1e4, 1e5
    stiffness = EA / (3 * length) * np.array([[7., -8., 1.], [-8., 16., -8.], [1., -8., 7.]]) if quadratic else EA / length * np.array([[1., -1.], [-1., 1.]])

    def forces(free):
        displacement = np.r_[free, 0.0]
        positions = np.column_stack((reference + displacement, np.zeros((nodes, 2))))
        line_energy, contact = _line_reference(positions, 0.11, np.zeros(3),
                                              np.array([0., 0., 1.]), k, length)
        gap = positions[0, 0] - 0.11  # The outward root cap has its sphere minimum.
        contact[0, 0] += ke * min(gap, 0.0)
        return stiffness @ displacement + contact[:, 0], contact, line_energy, 0.5 * ke * min(gap, 0.0)**2

    if prescribed is None:
        if quadratic:
            solution = root(lambda free: forces(free)[0][:-1] + np.array([0.2, 0.0]),
                            [-0.002002, -0.001001], tol=1e-11)
            assert solution.success, solution.message
            free = solution.x
        else:
            # Independent closed bar equation, including its distributed contact.
            def balance(delta):
                current_length = length + 0.002 + delta
                return EA / length * (0.002 + delta) + ke * delta + k * length * (
                    delta**2 / (2 * current_length) - delta**3 / (6 * current_length**2)) - 0.2
            delta = brentq(balance, 0.0, 1e-5, xtol=1e-18)
            free = np.array([-0.002 - delta])
    elif quadratic:
        middle = brentq(lambda value: forces([prescribed, value])[0][1], prescribed, 0.0, xtol=1e-18)
        free = np.array([prescribed, middle])
    else:
        free = np.array([prescribed])
    force, contact, line, cap = forces(free)
    if quadratic:
        assert_allclose(force[1], 0, atol=2e-11)
    return np.r_[free, 0.0], -force[0], force[-1], -contact[:, 0].sum(), line, cap


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("control", ["force", "displacement"])
def test_public_axial_equilibrium_reaction_work_and_gap(quadratic, control):
    model, definition, settings, load = _bar_case(quadratic)
    prescribed = -0.002005 if control == "displacement" else None
    options = {} if prescribed is None else {"control": "displacement", "displacement_control":
              DisplacementControl(node_id=1, dof="ux", target_displacement=prescribed)}
    result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
                                   num_steps=4, max_iterations=20, tolerance=1e-10, **options)
    assert result.status == "completed", result.failure_reason
    expected, press, reaction, beam_force, line, cap = _bar_reference(quadratic, prescribed)
    axial_dofs = [model.mesh.dof_manager.get_node_dofs(i)[0] for i in range(1, len(expected) + 1)]
    assert_allclose(result.displacements[axial_dofs], expected, rtol=1e-6, atol=1e-10)
    assert_allclose(0.2 * result.load_factor, press, rtol=2e-6, atol=2e-9)
    info = result.info["general_contact"]
    assert info["profile"] == PROFILE and info["active_endpoints"] == 1
    assert info["geometry_kinematics"] == "exact_literal_axis_projection_affine_static_v1"
    assert info["minimum_gap"] >= -1e-5
    assert info["endpoint_fields"][0]["branch"] == "sphere"
    assert_allclose(info["endpoint_fields"][0]["gap"], 0.002 + expected[0], atol=1e-10)
    assert line > 0 and cap > 0
    assert_allclose(info["line_contact_energy"], line, rtol=2e-5, atol=1e-16)
    assert_allclose(info["endpoint_contact_energy"], cap, rtol=2e-6, atol=1e-12)
    assert_allclose(info["algorithmic_contact_potential"], line + cap, rtol=2e-6, atol=1e-12)
    assert_allclose(info["beam_force"][0], beam_force, rtol=2e-6, atol=2e-9)
    assert_allclose(result.steps[-1].support_reactions["root"][0], reaction, rtol=2e-6, atol=2e-9)
    assert_allclose(np.asarray(info["beam_force"]) + info["rigid_force"], 0, atol=2e-12)
    actual_reaction=np.asarray(result.steps[-1].support_reactions["root"])
    actual_applied=.2*result.load_factor
    actual_force=np.asarray(info["beam_force"])+actual_reaction[:3]+(-actual_applied,0.,0.)
    assert np.linalg.norm(actual_force)/actual_applied<=1e-6
    assert np.linalg.norm(np.asarray(info["beam_moment"])+actual_reaction[3:])/(actual_applied*.2)<=1e-6
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('axial_'+('B3' if quadratic else 'B2')+'_'+control+'.json').write_text(
        json.dumps(dict(actual_u=result.displacements[axial_dofs].tolist(),reference_u=expected.tolist(),
            applied_force=actual_applied,reference_applied_force=press,
            reference_reaction=reaction,reference_contact_force=beam_force,
            actual_reaction=actual_reaction.tolist(),force_balance=actual_force.tolist(),info=info),
            indent=2),encoding='utf-8')


@pytest.mark.parametrize("quadratic", [False, True])
def test_fresh_split_restart_and_incompatible_cap_settings(quadratic):
    model, definition, settings, load = _bar_case(quadratic)
    common = dict(contact=definition, contact_settings=settings, num_steps=4,
                  max_iterations=20, tolerance=1e-10, emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, load, **common)
    assert full.status == "completed", full.failure_reason
    model, definition, settings, load = _bar_case(quadratic)
    prefix = solve_static_nonlinear(model, load, **{**common, "num_steps": 2}, max_load_factor=0.75)
    assert prefix.status == "completed", prefix.failure_reason
    checkpoint = json.loads(json.dumps(prefix.to_restart_checkpoint(), allow_nan=False))
    model, definition, settings, load = _bar_case(quadratic)
    resumed = solve_static_nonlinear(model, load, **{**common, "contact": definition,
        "contact_settings": settings, "num_steps": 2}, restart_checkpoint=checkpoint)
    assert resumed.status == "completed", resumed.failure_reason
    assert_allclose(resumed.displacements, full.displacements, rtol=1e-6, atol=1e-10)
    for incompatible in (replace(settings, beam_end_cap_work="owned_point"),
                         replace(settings, maximum_swept_subdivisions=settings.maximum_swept_subdivisions + 1)):
        with pytest.raises(ValueError, match="restart|incompatible|contract"):
            solve_static_nonlinear(model, load, **{**common, "contact_settings": incompatible},
                                   restart_checkpoint=checkpoint)


@pytest.mark.parametrize("cancel", [False, True])
def test_actual_candidate_interruption_rolls_back_and_replays_fresh(cancel, monkeypatch):
    model, definition, settings, load = _bar_case()
    common = dict(contact=definition, contact_settings=settings, num_steps=4,
                  max_iterations=20, tolerance=1e-10, emit_restart_checkpoint=True)
    token = CancellationToken()
    armed, raised = [False], [False]
    original = HemisphereCylinderManager.evaluate

    def interrupt(event):
        if event.event_type == "nonlinear_static_step" and event.metadata["load_factor"] >= 0.5:
            if cancel:
                token.cancel("cylinder cap owned interruption")
            else:
                armed[0] = True

    def fail_after_candidate(self, *values, **options):
        candidate = original(self, *values, **options)
        if armed[0] and not raised[0]:
            raised[0] = True
            assert self._geometry_cache is None
            raise RuntimeError("cylinder cap owned interruption")
        return candidate

    with monkeypatch.context() as patch:
        if not cancel:
            patch.setattr(HemisphereCylinderManager, "evaluate", fail_after_candidate)
        with pytest.raises(SolveCancelled if cancel else RuntimeError, match="cylinder cap owned interruption"):
            solve_static_nonlinear(model, load, **common, progress_callback=interrupt, cancellation_token=token)
    assert armed[0] or token.is_cancelled
    if not cancel:
        assert raised[0]
    replay = solve_static_nonlinear(model, load, **common)
    fresh, definition2, settings2, load2 = _bar_case()
    reference = solve_static_nonlinear(fresh, load2, **{**common, "contact": definition2, "contact_settings": settings2})
    assert replay.status == reference.status == "completed"
    np.testing.assert_array_equal(replay.displacements, reference.displacements)
    assert replay.to_restart_checkpoint() == reference.to_restart_checkpoint()


def test_explicit_profile_fixed_descriptor_and_no_application_admission():
    model, definition, settings = _support_case(True)
    assert validate_general_contact(model, definition, settings).profile == PROFILE
    assert validate_general_contact(model, definition, settings).valid
    capability = report_general_contact_capability(model, definition, settings)
    assert capability.implemented and not capability.admitted_for_application
    assert not capability.tested and not capability.independently_accepted and not capability.released
    assert validate_general_contact(model, definition, replace(settings, beam_end_cap_work="owned_point")).profile == LEGACY_PROFILE
    moving = replace(definition, rigid_bodies=(replace(definition.rigid_bodies[0], path_translation=(0.001, 0., 0.)),))
    from anyfesolver.general_contact.beam_hemisphere_cylinder_motion import PROFILE as MOVING_PROFILE
    assert validate_general_contact(model, moving, settings).profile == MOVING_PROFILE
    assert validate_general_contact(model, moving, settings).valid
    with pytest.raises(ValueError, match="fixed infinite cylinder|unsupported"):
        HemisphereCylinderManager(model, moving, settings)
    with pytest.raises(FrozenInstanceError):
        definition.rigid_bodies[0].point = (1., 0., 0.)
    for unsupported in ("free_cylinder", "capped_cylinder"):
        with pytest.raises(ValueError):
            RigidContactBody("unsupported", unsupported, (0., 0., 0.), normal=(0., 0., 1.), radius=0.1)


def test_true_end_topology_excludes_midnode_of_another_beam():
    model, definition, settings = _support_case(True)
    assert set(HemisphereCylinderManager(model, definition, settings).endpoints) == {1, 3}
    model.add_node(10, 0.112, -0.1, 0.)
    model.add_node(11, 0.112, 0.1, 0.)
    model.add_element(2, QuadraticBeamElement(2, (10, 1, 11), "elastic",
        {"area": 1e-3, "Iy": 1e-4, "Iz": 1e-4, "J": 1e-4}))
    assert set(HemisphereCylinderManager(model, definition, settings).endpoints) == {3}


def test_active_nonunique_cap_query_rejects_and_discards_trial_cache():
    model, definition, settings = _support_case()
    manager = HemisphereCylinderManager(model, definition, settings)
    initial = np.zeros(model.mesh.dof_manager.total_dofs)
    reference = manager.evaluate(initial)
    center = np.array([0.1001, 0., 0.])
    direction = np.array([0.9989, math.sqrt(1 - 0.9989**2), 0.])
    trial = initial.copy()
    for node, point in ((1, center), (2, center - 1e-5 * direction)):
        trial[model.mesh.dof_manager.get_node_dofs(node)[:3]] = point - model.mesh.nodes[node].coords()
    with pytest.raises(ValueError, match="nonunique|multiple|ambiguous"):
        manager.evaluate(trial)
    assert manager._geometry_cache is None
    fresh = manager.evaluate(initial)
    np.testing.assert_array_equal(fresh[0], reference[0])
    assert_allclose(fresh[1].toarray(), reference[1].toarray(), atol=0, rtol=0)
    assert fresh[2].to_dict() == reference[2].to_dict()
