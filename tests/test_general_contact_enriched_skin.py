"""Actual enriched geometry/work/layout/state development controls.

No baseline oracle is altered and no full174/300 or cut-uniform qualification
is claimed.  The exact full-cell polynomial traction oracle is independent
of the production overlap differentiation and stationary elimination.
"""
import json
import numpy as np
import pytest

from examples.general_contact_enriched_patch import build
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_static import MixedCriteria, IterationControl


@pytest.fixture(scope="module")
def touch():
    model, system = build(target_z=.02, source_offset=.01, target_offset=.01)
    u = np.zeros(system.offset.size)
    z = np.ones(4)
    raw = system.linearize(u, z)
    return model, system, u, z, raw


def test_exact_full_cell_constant_affine_and_all_four_q1_traction_moments(touch, record_property):
    _, system, u, z, raw = touch
    assert system.offset.size == system.physical_dofs+4
    assert system.transform.shape[1] == 4  # beta is EXPLICIT even with every node fixed
    assert system.pressure_ids == (1, 2, 3, 4)
    assert np.allclose(raw.support, .25, rtol=0, atol=1e-10)
    assert np.max(np.abs(raw.normalized_gap)) < 1e-12
    nodal = np.array(((1, -1, -1, 1), (1, 1, -1, -1),
                      (1, 1, 1, 1), (1, -1, 1, -1)), dtype=float)
    # Each column is one pressure polynomial 1,r,s,rs.  Exact independent
    # integration on [-1,1]^2, area1, gives the diagonal below.  Signed columns
    # are virtual pressure probes, not accepted tensile-contact states.
    work = (raw.support[:, None]*nodal).T @ raw.gap_gradient[:, system.physical_dofs:]
    assert np.allclose(work, np.diag((4/9, 4/45, 4/45, 4/225)), rtol=1e-8, atol=1e-10)
    assert np.allclose(raw.force, -raw.gap_gradient.T @ z, rtol=0, atol=0)
    assert np.linalg.eigvalsh(raw.pressure_gram)[0] > 0  # this full-cell case only
    record_property("full_cell_work_matrix", json.dumps(work.tolist()))
    record_property("full_cell_pressure_gram_eigenvalues", json.dumps(np.linalg.eigvalsh(raw.pressure_gram).tolist()))


def test_complete_mixed_tangent_includes_all_added_coordinates_and_offsets(touch, record_property):
    _, system = build(target_z=.02, source_offset=.01, target_offset=.01, fixed_nodal=False)
    z = np.ones(4)
    q = np.zeros(system.transform.shape[1])
    _, raw, r, c, jac = system.equations(q, z, np.zeros(system.offset.size))
    direction = np.linspace(-.2, .3, q.size+4)
    direction[q.size-4:q.size] = (.3, -.5, .7, .2)
    expected = jac @ direction
    errors = []
    for h in (1e-7, 3e-8):
        plus = system.equations(q+h*direction[:q.size], z+h*direction[q.size:], np.zeros(system.offset.size))
        minus = system.equations(q-h*direction[:q.size], z-h*direction[q.size:], np.zeros(system.offset.size))
        assert plus[1].branches == minus[1].branches == raw.branches
        actual = (np.concatenate((plus[2], plus[3]))-np.concatenate((minus[2], minus[3])))/(2*h)
        errors.append(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert max(errors) <= 1e-5
    record_property("two_h_relative_errors", json.dumps(errors))


def test_cut_front_keeps_every_q1_mode_and_matches_independent_physical_work(record_property):
    _, system = build(cut_fraction=.05)
    raw = system.linearize(np.zeros(system.offset.size), np.ones(4))
    nodal = np.array(((1, -1, -1, 1), (1, 1, -1, -1),
                      (1, 1, 1, 1), (1, -1, 1, -1)), dtype=float)
    actual = (raw.support[:, None]*nodal).T @ raw.gap_gradient[:, system.physical_dofs:]
    expected = np.zeros((4, 4))
    x, w = np.polynomial.legendre.leggauss(6)
    for a, wa in zip(x, w):
        r = .95+.05*a  # natural cut r in [.9,1], physical strip width .05
        for s, ws in zip(x, w):
            p = np.array((1., r, s, r*s))
            expected += .25*.05*wa*ws*(1-r*r)*(1-s*s)*np.outer(p, p)
    assert system.pressure_ids == (1, 2, 3, 4) and np.all(raw.support > 0)
    assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10)
    assert np.linalg.eigvalsh((actual+actual.T)/2)[0] > 0
    record_property("cut_work_matrix", json.dumps(actual.tolist()))
    record_property("cut_work_eigenvalues", json.dumps(np.linalg.eigvalsh((actual+actual.T)/2).tolist()))
    # This one smooth strip proves local mode observability, NOT a uniform
    # inf-sup bound, full pressure response, or acceptance of all cut branches.


def test_nonzero_bubble_rigid_translation_rotation_work_and_both_body_offsets(touch):
    model, system, u0, z, _ = touch
    u = u0.copy()
    u[-4:] = (2e-4, 1e-4, -7e-5, 9e-5)
    raw = system.linearize(u, z)
    translation = np.zeros_like(u)
    rotation = np.zeros_like(u)
    omega = np.array((.2, -.3, .1))
    for node, reference in model.mesh.nodes.items():
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        position = np.array((reference.x, reference.y, reference.z))
        translation[np.asarray(dofs[:3])] = (.3, -.1, .2)
        rotation[np.asarray(dofs[:3])] = np.cross(omega, position)
        rotation[np.asarray(dofs[3:])] = omega
    # beta is unchanged by a rigid body rotation: current midsurface frame
    # rotates its actual physical field.  No beta force is counted as net force.
    for direction in (translation, rotation):
        assert abs(raw.force @ direction) <= 1e-8*max(
            np.linalg.norm(raw.force)*np.linalg.norm(direction), 1e-12)
    source_force = np.array([sum(raw.force[model.mesh.dof_manager.get_node_dofs(i)[k]]
                                 for i in (1, 2, 3, 4)) for k in range(3)])
    target_force = np.array([sum(raw.force[model.mesh.dof_manager.get_node_dofs(i)[k]]
                                 for i in (5, 6, 7)) for k in range(3)])
    assert np.linalg.norm(source_force+target_force) <= 1e-8*max(np.linalg.norm(source_force), 1e-12)


def test_midsurface_bubble_is_independent_of_top_bottom_side(touch):
    _, system, u, _, _ = touch
    from dataclasses import replace
    pair = system.pairs[0]
    bottom = replace(pair, definition=replace(pair.definition, source_offset=-.01))
    u = u.copy()
    u[3:6] = (.02, -.01, .03)
    u[-4:] = (1e-4, 2e-5, -3e-5, 4e-5)
    top_x, top_a, top_h = pair.map(u)
    bottom_x, bottom_a, bottom_h = bottom.map(u)
    assert np.array_equal(top_x[21:], bottom_x[21:])
    assert np.array_equal(top_a[21:], bottom_a[21:])
    assert np.array_equal(top_h[21:], bottom_h[21:])
    assert not np.array_equal(top_x[:12], bottom_x[:12])


def test_actual_owner_checkpoint_cancel_refusal_and_definition_identity_are_atomic(record_property):
    _, system = build(target_shift=20.)  # genuine no-overlap cheap accepted prefix
    criteria = MixedCriteria(.04, 1e-6, 1e-8)
    control = IterationControl(4, 4)
    result = system.solve(np.zeros(system.offset.size), criteria=criteria, control=control)
    assert result["status"] == "DEVELOPMENT_NCP_CONVERGED"
    # Independent analytic reference-linear bubble response on the unit square:
    # int f0^2=16/15,int f1^2=16/105,int f0'^2=8/3,int f1'^2=8/5.
    # The physical derivative factors2 cancel reference Jacobian1/4.
    shear = (5/6)*(1e5/(2*1.3))*.05
    independent = shear*np.diag((256/45, 3328/1575, 3328/1575, 256/525))
    desired = np.array((1e-7, -2e-7, 1.5e-7, -.5e-7))
    load = np.zeros(system.offset.size)
    load[-4:] = independent @ desired
    response = system.solve(load, criteria=criteria, control=control)
    assert np.allclose(system._accepted.coordinates[-4:], desired, rtol=1e-8, atol=1e-14)
    assert response["equilibrium"] <= 1e-6 and response["complementarity"] <= 1e-8
    record_property("independent_open_response_beta", json.dumps(system._accepted.coordinates[-4:].tolist()))
    record_property("independent_open_response_metrics", json.dumps(
        dict(equilibrium=response["equilibrium"], complementarity=response["complementarity"],
             counts=response["counts"])))
    before = system.checkpoint()
    accepted = system._accepted
    for malformed in (b"{", b"not-json", b"\xff", b"[]", b'{"coordinates":NaN}'):
        with pytest.raises(EnrichedProfileError):
            system.restore(malformed, criteria=criteria)
        assert system.checkpoint() == before and system._accepted is accepted
    for malformed_array in (["bad"], {"0": 1.}, None):
        payload = json.loads(before)
        payload["coordinates"] = malformed_array
        with pytest.raises(EnrichedProfileError):
            system.restore(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
                           criteria=criteria)
        assert system.checkpoint() == before and system._accepted is accepted
    with pytest.raises(SolveCancelled):
        system.solve(np.zeros(system.offset.size), criteria=criteria, control=control, cancel=lambda: True)
    assert system.checkpoint() == before and system._accepted is accepted
    overflow = json.loads(before)
    overflow["coordinates"][-1] = 1e308
    overflow["reduced"][-1] = 1e308
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(EnrichedProfileError, match="nonfinite"):
            system.restore(json.dumps(overflow, sort_keys=True, separators=(",", ":")).encode(),
                           criteria=criteria)
    assert system.checkpoint() == before and system._accepted is accepted
    calls = []
    def cancel_after_pair():
        calls.append(1)
        return len(calls) == 3
    with pytest.raises(SolveCancelled):
        system.solve(np.zeros(system.offset.size), criteria=criteria, control=control, cancel=cancel_after_pair)
    assert len(calls) == 3 and system.checkpoint() == before and system._accepted is accepted
    for mutation in (dict(profile="legacy_bilinear"), dict(definition="0"*64), dict(generation=0),
                     dict(profile="E4PL_PHYSICAL_B4_REFERENCE_LINEAR_V1")):
        payload = json.loads(before)
        payload.update(mutation)
        with pytest.raises(EnrichedProfileError):
            system.restore(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), criteria=criteria)
        assert system.checkpoint() == before and system._accepted is accepted
    payload = json.loads(before)
    payload["coordinates"][-1] = 1e-3
    payload["reduced"][-1] = 1e-3
    with pytest.raises(EnrichedProfileError, match="residual"):
        system.restore(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), criteria=criteria)
    assert system.checkpoint() == before and system._accepted is accepted
    _, other = build(target_shift=20.)
    other.restore(before, criteria=criteria)
    assert other.checkpoint() == before


def test_geometry_nonfinite_and_quadrature_cancel_refuse_without_publish(touch):
    _, system, u, z, _ = touch
    before = system._accepted
    bad = u.copy()
    bad[-1] = np.nan
    with pytest.raises(EnrichedProfileError):
        system.linearize(bad, z)
    with pytest.raises(SolveCancelled):
        system.linearize(u, z, cancel=lambda: True)
    assert system._accepted is before
