"""Equilibrated force recovery, distinct from loaded-pose qualification."""
from dataclasses import replace

import numpy as np
import pytest
from anyfesolver import stiffened_shell as api
from anyfesolver._b2_physical_flexibility_linear import LinearOperator
from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point
from anyfesolver.control import CancellationToken
from test_stiffened_shell_b2_body_acceleration import _loaded_definition, _mass


def _section(coupled=False):
    diagonal = np.diag((17., 8., 9., 3., 4., 5.))
    change = np.eye(6)
    if coupled:
        change[0, 4], change[1, 5], change[2, 3] = .2, -.3, .15
    return api.LinearSection.generalized(change.T@diagonal@change)


@pytest.mark.parametrize("coupled", (False, True))
def test_fixed_end_uniform_loading_has_nonzero_equilibrated_fields(coupled):
    length = 2.
    section = _section(coupled)
    operator = LinearOperator(length, section)
    wrench = np.array((.7, -1.2, .9, .2, -.3, .4))
    stations = np.linspace(0., 1., 9)
    fields = operator.recover_equilibrated_sections(np.zeros(12), uniform_wrench=wrench, stations=stations)
    assert np.any(fields.resultants_left != 0.)
    np.testing.assert_array_equal(operator.recover_sections(np.zeros(12)).resultants, 0.)
    np.testing.assert_array_equal(fields.resultants_left, fields.resultants_right)
    x = length*stations
    reference_force = fields.resultants_left[0, :3]-x[:, None]*wrench[:3]
    reference_moment = (fields.resultants_left[0, 3:]-x[:, None]*np.cross((1.,0.,0.), fields.resultants_left[0, :3])
                        -x[:, None]*wrench[3:]+.5*x[:, None]**2*np.cross((1.,0.,0.), wrench[:3]))
    np.testing.assert_allclose(fields.resultants_left[:, :3], reference_force, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fields.resultants_left[:, 3:], reference_moment, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fields.resultants_left@np.linalg.inv(section.stiffness).T,
                               fields.strains_left, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(fields.endpoint_tractions[:6], -fields.resultants_left[0], rtol=0., atol=0.)
    np.testing.assert_allclose(fields.endpoint_tractions[6:], fields.resultants_right[-1], rtol=0., atol=0.)
    # Independent strain integration proves zero end-to-end deformation for
    # the fixed-end loaded reconstruction, including generalized couplings.
    points, weights = np.polynomial.legendre.leggauss(5)
    result = operator.recover_equilibrated_sections(np.zeros(12), uniform_wrench=wrench,
                                                   stations=(points+1.)/2.)
    strain = result.strains_left
    xx = length*(points+1.)/2.
    integrated = length/2.*(weights@strain)
    rotation = integrated[3:]
    u_end = integrated[:3].copy()
    u_end[1] += length/2.*np.sum(weights*(length-xx)*strain[:, 5])
    u_end[2] -= length/2.*np.sum(weights*(length-xx)*strain[:, 4])
    np.testing.assert_allclose(np.r_[u_end, rotation], 0., atol=2e-15)
    for value in (fields.stations, fields.resultants_left, fields.strains_right, fields.endpoint_tractions):
        with pytest.raises(ValueError): value.setflags(write=True)


def test_interior_point_jumps_offset_work_and_endpoint_actions():
    definition = _loaded_definition("mixed")
    member = definition.members[0]
    vertices = np.array([definition.shell_nodes_m[n] for n in definition.shells[0].node_ids])
    offset = np.array((.04, -.03, .02))
    law = compile_reference_point(member, vertices, station=.5, offset_local_m=offset)
    multiplier = np.array((.4, -.2, .6, -.7, .3, .9))
    port = -law.beam_constraint_differential.T@multiplier
    local = np.kron(np.eye(2), member.local_frame_rows())@port
    local[3:] += np.cross(offset, local[:3])
    nodal_external = -law.forces(multiplier)[1]
    trace = member.local_operator().trace(np.zeros(12), stations=(.5,))
    np.testing.assert_allclose(member.local_dof_map().T@trace.jacobian[0].T@local,
                               nodal_external, rtol=2e-12, atol=2e-14)
    u = np.random.default_rng(7).normal(size=12)
    operator = member.local_operator()
    fields = operator.recover_equilibrated_sections(u, point_actions=((.5, local),), stations=(0., .5, 1.))
    np.testing.assert_allclose(fields.resultants_right[1]-fields.resultants_left[1], -local,
                               rtol=2e-13, atol=2e-14)
    endpoint = fields.endpoint_tractions.reshape(2, 6)
    np.testing.assert_allclose(endpoint[:, :3].sum(axis=0)+local[:3], 0., atol=2e-12)
    moment = (endpoint[:, 3:].sum(axis=0)+np.cross((member.length_m,0.,0.), endpoint[1,:3])
              +np.cross((.5*member.length_m,0.,0.),local[:3])+local[3:])
    np.testing.assert_allclose(moment, 0., atol=2e-12)
    with pytest.raises(ValueError, match="interior"):
        operator.recover_equilibrated_sections(u, point_actions=((0., local),))
    with pytest.raises(ValueError): law.beam_constraint_differential.setflags(write=True)


def test_combined_distributed_and_point_fields_preserve_endpoint_compatibility():
    operator = LinearOperator(1.7, _section(True))
    u = np.random.default_rng(13).normal(size=12)
    wrench = np.array((.6, -.4, .3, -.5, .7, .8))
    actions = ((.35, np.array((.8, -.2, -.7, .1, .5, -.3))),
               (.6, np.array((-.2, .3, .6, -.4, .2, .1))))
    points, weights = np.polynomial.legendre.leggauss(6)
    sums = np.zeros(6)
    bend = np.zeros(2)
    for a,b in zip((0., .35, .6), (.35, .6, 1.)):
        stations = a+(b-a)*(points+1.)/2.
        fields = operator.recover_equilibrated_sections(u, uniform_wrench=wrench,
                                                       point_actions=actions, stations=stations)
        weighted = 1.7*(b-a)/2.*weights
        sums += weighted@fields.strains_left
        bend += (weighted*(1.7-1.7*stations))@fields.strains_left[:, (4,5)]
    end = np.r_[u[:3]+sums[:3], u[3:6]+sums[3:]]
    end[1] += 1.7*u[5]+bend[1]
    end[2] -= 1.7*u[4]+bend[0]
    np.testing.assert_allclose(end, u[6:], rtol=2e-13, atol=2e-13)


def test_typed_footprint_batches_recover_only_selected_members():
    from test_stiffened_shell_b2_footprint_reference_modes import _definition
    definition = _definition(cells=5, points=True)
    definition = replace(definition, members=tuple(replace(m, mass=_mass()) for m in definition.members))
    analysis = api.create_analysis(definition=definition)
    try:
        body = analysis.body_acceleration_load((.1, -.2, -.3))
        single = analysis.solve_linear(body)
        assert single.status == "completed", single.failure_reason
        assert isinstance(single, api.B2BodyLoadedFootprintResult)
        assert len(single.equilibrated_sections) == len(definition.members)
        member = definition.members[-1].member_id
        result = analysis.solve_linear_many((("one",body),("two",body)), selected_dofs=(0,),
            recover_members={member: (0., .5, 1.)})
        assert result.status == "completed" and result.factorization_count == 1
        assert isinstance(result.cases[0], api.B2BodyLoadedFootprintSelectedCase)
        assert len(result.cases[0].equilibrated_sections) == 1
    finally:
        analysis.close()


def test_endpoint_points_remain_nodal_and_preserve_existing_resolution_failure():
    from test_stiffened_shell_b2_footprint_reference_modes import _definition
    definition = _definition(cells=5, points=True)
    definition = replace(definition, points=tuple(replace(p, station=1.) for p in definition.points))
    analysis = api.create_analysis(definition=definition)
    try:
        assert analysis._section_point_actions == {}
        body = analysis.body_acceleration_load((.1, -.2, -.3))
        raw, typed = analysis.solve_linear(body.load), analysis.solve_linear(body)
        assert raw.status == typed.status == "failed"
        assert raw.failure_reason == typed.failure_reason == "panel mixed-load resolution insufficient"
    finally:
        analysis.close()


def test_typed_batch_detects_metadata_drift_even_when_load_bytes_do_not_change():
    analysis = api.create_analysis(definition=_loaded_definition("mixed"))
    try:
        first = analysis.body_acceleration_load((0., 0., 0.))
        later = analysis.body_acceleration_load((0., 0., 0.))
        # Deliberate corruption bypasses the frozen record; signed zero leaves
        # the applied vector equal but changes the submitted provenance bytes.
        def corrupt(_):
            object.__setattr__(later, "acceleration_m_s2", (0., -0., 0.))
        with pytest.raises(ValueError, match="changed after submission"):
            analysis.solve_linear_many((("first",first),("corrupted",later)),
                selected_dofs=(49,50), recover_members={}, on_case=corrupt)
        assert analysis._handle.solve_count == 1
    finally:
        analysis.close()


@pytest.mark.parametrize("kind", ("q4", "mixed", "s3"))
def test_typed_body_load_preserves_legacy_state_and_recovers_equilibrium(kind):
    definition = _loaded_definition(kind)
    analysis = api.create_analysis(definition=definition)
    same_model = api.create_analysis(definition=definition)
    try:
        body = analysis.body_acceleration_load((.3, -.4, -.7))
        legacy = analysis.solve_linear(body.load)
        loaded = analysis.solve_linear(body)
        assert legacy.status == loaded.status == "completed"
        assert isinstance(loaded, api.B2BodyLoadedPanelResult)
        assert not hasattr(legacy, "equilibrated_sections")
        assert loaded.displacements == legacy.displacements
        assert loaded.point_multipliers == legacy.point_multipliers
        assert loaded.beam_resultants_local == legacy.beam_resultants_local
        assert loaded.internal_strain_energy == legacy.internal_strain_energy
        assert loaded.body_model_identity == body.model_identity
        assert same_model.solve_linear(body).status == "completed"
        assert len(loaded.equilibrated_sections) == 2
        for fields in loaded.equilibrated_sections:
            assert fields.stations == (0., .5, 1.) and not fields.production_qualified
            np.testing.assert_allclose(fields.endpoint_tractions_local[:6], -np.array(fields.resultants_left[0]), atol=0.)
            np.testing.assert_allclose(fields.endpoint_tractions_local[6:], fields.resultants_right[-1], atol=0.)
            assert np.any(np.array(fields.resultants_left[1]) != np.array(fields.resultants_right[1]))
        batch = analysis.solve_linear_many((("a", body), ("b", body)), selected_dofs=(49,50),
                                           recover_members={"beam-left": (0., .5, 1.)})
        assert batch.status == "completed" and batch.factorization_count == 1
        assert len(batch.cases[0].equilibrated_sections) == 1
        token = CancellationToken()
        def stop(_): token.cancel()
        cancelled = analysis.solve_linear_many((("first",body),("never",body)), selected_dofs=(49,50),
            recover_members={"beam-left": (.5,)}, cancellation_token=token, on_case=stop)
        assert cancelled.status == "cancelled" and len(cancelled.cases) == 1
        bad = replace(body, load=body.load*2.)
        before = analysis._handle.solve_count
        with pytest.raises(ValueError, match="disagree"):
            analysis.solve_linear(bad)
        foreign = replace(body, model_identity="wrong")
        with pytest.raises(ValueError, match="different compiled model"):
            analysis.solve_linear_many((("wrong",foreign),), selected_dofs=(49,50))
        assert analysis._handle.solve_count == before
        changed_definition = replace(definition, members=tuple(replace(m,
            section=api.LinearSection.generalized(np.array(m.section.stiffness)*1.01)) for m in definition.members))
        changed = api.create_analysis(definition=changed_definition)
        try:
            with pytest.raises(ValueError, match="different compiled model"):
                changed.solve_linear(body)
        finally:
            changed.close()
    finally:
        analysis.close(); same_model.close()
