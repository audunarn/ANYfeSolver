"""Reference body-load work and physical wrench; no gravity qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import ExperimentalB2Q4PanelError
from anyfesolver._b2_physical_flexibility_linear import RepresentabilityError
from test_stiffened_shell_b2_mixed_panel_experimental import _definition


MASS = .3
ORIGIN = np.array((.07, .04, -.03))
CENTER = np.array((.02, -.015, .025))


def _mass(origin=ORIGIN, center=CENTER, inertia=1.):
    cross = np.array(((0., -center[2], center[1]),
                      (center[2], 0., -center[0]),
                      (-center[1], center[0], 0.)))
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = MASS*np.eye(3)
    matrix[:3, 3:] = -MASS*cross
    matrix[3:, :3] = matrix[:3, 3:].T
    matrix[3:, 3:] = inertia*np.diag((.003, .002, .0025)) + MASS*(center@center*np.eye(3)-np.outer(center, center))
    return api.MassMetadata("physical-body-mass", tuple(origin), matrix)


def _loaded_definition(kind="mixed", distorted=False):
    definition, _ = _definition(kind)
    nodes = dict(definition.shell_nodes_m)
    if distorted:
        nodes[5] = (.9, 1.15, 0.)
    return replace(definition, shell_nodes_m=nodes,
        shells=tuple(replace(s, density_kg_m3=2.) for s in definition.shells),
        members=tuple(replace(m, mass=_mass()) for m in definition.members))


def _shell_quadrature(vertices):
    if len(vertices) == 3:
        area = np.linalg.norm(np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]))/2.
        return [(np.ones(3)/3., area)]
    points, weights = np.polynomial.legendre.leggauss(5)
    out = []
    for x, wx in zip(points, weights):
        for y, wy in zip(points, weights):
            n = .25*np.array(((1-x)*(1-y), (1+x)*(1-y), (1+x)*(1+y), (1-x)*(1+y)))
            dx = .25*np.array((-(1-y), 1-y, 1+y, -(1+y)))
            dy = .25*np.array((-(1-x), -(1+x), 1+x, 1-x))
            out.append((n, wx*wy*np.linalg.norm(np.cross(dx@vertices, dy@vertices))))
    return out


def _oracle(definition, acceleration):
    """Direct action integration: no candidate mass matrix or owner load map."""
    shell_ids = sorted(definition.shell_nodes_m)
    beam_ids = sorted({n for m in definition.members for n in m.node_ids})
    shell_index = {n:i for i,n in enumerate(shell_ids)}
    beam_index = {n:i+len(shell_ids) for i,n in enumerate(beam_ids)}
    load = np.zeros(6*(len(shell_ids)+len(beam_ids)))
    force, moment = np.zeros(3), np.zeros(3)
    for shell in definition.shells:
        xyz = np.array([definition.shell_nodes_m[n] for n in shell.node_ids])
        for n, area in _shell_quadrature(xyz):
            f = shell.density_kg_m3*shell.thickness_m*area*acceleration
            force += f
            moment += np.cross(n@xyz, f)
            for node, coefficient in zip(shell.node_ids, n):
                load[6*shell_index[node]:6*shell_index[node]+3] += coefficient*f
    points, weights = np.polynomial.legendre.leggauss(8)
    for member in definition.members:
        rotation = member.local_frame_rows()
        local_force = MASS*(rotation@acceleration)
        # Independent physical first moment, including the declared mass origin.
        local_wrench = np.r_[local_force, np.cross(ORIGIN+CENTER, local_force)]
        trace = member.local_operator().trace(np.zeros(12), stations=tuple((points+1.)/2.))
        local = sum(member.length_m*w/2.*(j.T@local_wrench) for w,j in zip(weights, trace.jacobian))
        slots = [6*beam_index[n]+d for n in member.node_ids for d in range(6)]
        load[slots] += member.local_dof_map().T@local
        f = member.length_m*MASS*acceleration
        centroid = np.mean(member.reference_points_m, axis=0)+rotation.T@(ORIGIN+CENTER)
        force += f
        moment += np.cross(centroid, f)
    xyz = np.array([definition.shell_nodes_m[n] for n in shell_ids]+[
        next(m.reference_points_m[m.node_ids.index(n)] for m in definition.members if n in m.node_ids)
        for n in beam_ids])
    return load, force, moment, xyz


@pytest.mark.parametrize("kind", ("q4", "mixed", "s3"))
def test_physical_body_load_matches_direct_work_and_global_wrench(kind):
    definition = _loaded_definition(kind, distorted=True)
    acceleration = np.array((.4, -.2, -1.5))
    expected, force, moment, xyz = _oracle(definition, acceleration)
    analysis = api.create_analysis(definition=definition)
    try:
        before = analysis._handle.solve_count
        body = analysis.body_acceleration_load(acceleration)
        assert isinstance(body, api.B2ReferenceBodyLoad) and not body.production_qualified
        assert "particular" in body.member_recovery
        np.testing.assert_allclose(body.load, expected, rtol=2e-12, atol=2e-14)
        poses = body.load.reshape(-1, 6)
        np.testing.assert_allclose(poses[:, :3].sum(axis=0), force, rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(np.cross(xyz, poses[:, :3]).sum(axis=0)+poses[:, 3:].sum(axis=0), moment, rtol=2e-12, atol=2e-14)
        virtual = np.random.default_rng(11).normal(size=body.load.shape)
        assert virtual@body.load == pytest.approx(virtual@expected, rel=2e-12, abs=2e-14)
        assert analysis._handle.solve_count == before and analysis._handle.factorization_count == 1
        if kind != "q4":
            assert analysis._M is None and "modes" not in analysis.capabilities.operations
        assert "body_acceleration_load" in analysis.capabilities.operations
        acceleration[:] = 100.
        assert body.acceleration_m_s2 == (.4, -.2, -1.5)
        with pytest.raises(ValueError): body.load.setflags(write=True)
    finally:
        analysis.close()


def test_oblique_frame_and_reversed_members_preserve_physical_body_load():
    definition = _loaded_definition("q4")
    rotation = Rotation.from_rotvec((.2, -.1, .3)).as_matrix()
    origin = np.array((1.3, -.5, .8))
    transformed = replace(definition,
        shell_nodes_m={n:tuple(rotation@np.asarray(p)+origin) for n,p in definition.shell_nodes_m.items()},
        shells=tuple(replace(s, reference_normal=tuple(rotation@np.asarray(s.reference_normal))) for s in definition.shells),
        members=tuple(replace(m, reference_points_m=tuple(tuple(rotation@np.asarray(p)+origin) for p in m.reference_points_m),
                              local_y_global=tuple(rotation@np.asarray(m.local_y_global))) for m in definition.members))
    acceleration = np.array((-.2, .7, -1.1))
    expected, force, moment, xyz = _oracle(transformed, acceleration)
    reversed_members = []
    for m in transformed.members:
        reversal = np.diag((-1., 1., -1.))
        reversed_members.append(replace(m, node_ids=m.node_ids[::-1], reference_points_m=m.reference_points_m[::-1],
            mass=_mass(reversal@ORIGIN, reversal@CENTER)))
    for model in (transformed, replace(transformed, members=tuple(reversed_members))):
        analysis = api.create_analysis(definition=model)
        try:
            np.testing.assert_allclose(analysis.body_acceleration_load(acceleration).load, expected, rtol=3e-12, atol=3e-14)
        finally:
            analysis.close()


def test_shifted_mass_origin_and_large_rotary_inertia_do_not_change_weight():
    definition = _loaded_definition("q4")
    expected, *_ = _oracle(definition, np.array((.4, -.2, -1.5)))
    shift = np.array((.05, -.04, .03))
    for mass in (_mass(ORIGIN+shift, CENTER-shift), _mass(inertia=1e16)):
        model = replace(definition, members=tuple(replace(m, mass=mass) for m in definition.members))
        analysis = api.create_analysis(definition=model)
        try:
            np.testing.assert_allclose(analysis.body_acceleration_load((.4, -.2, -1.5)).load, expected, rtol=3e-12, atol=3e-14)
        finally:
            analysis.close()


def test_rigid_footprint_and_points_add_no_attachment_mass():
    from test_stiffened_shell_b2_footprint_reference_modes import _definition as footprint_definition
    definition = footprint_definition(cells=5, points=True)
    definition = replace(definition, members=tuple(replace(m, mass=_mass()) for m in definition.members))
    expected, *_ = _oracle(definition, np.array((.4, -.2, -1.5)))
    analysis = api.create_analysis(definition=definition)
    try:
        np.testing.assert_allclose(analysis.body_acceleration_load((.4, -.2, -1.5)).load, expected, rtol=2e-12, atol=2e-14)
        assert "body_acceleration_load" in analysis.capabilities.operations
        result = analysis.solve_linear(expected, recover_members={})
        assert result.status == "completed", result.failure_reason
    finally:
        analysis.close()


def test_unrepresentable_body_force_is_not_returned_as_a_success():
    definition = _loaded_definition("s3")
    definition = replace(definition, shells=tuple(replace(s, density_kg_m3=5e307) for s in definition.shells))
    analysis = api.create_analysis(definition=definition)
    try:
        with pytest.raises(RepresentabilityError, match="not representable"):
            analysis.body_acceleration_load((1e20, 0., 0.))
        assert analysis._handle.solve_count == 0
    finally:
        analysis.close()


@pytest.mark.parametrize("tiny", (np.nextafter(0., 1.), 2.*np.nextafter(0., 1.)))
def test_underflow_cannot_discard_required_body_force(tiny):
    analysis = api.create_analysis(definition=_loaded_definition("mixed"))
    try:
        with pytest.raises(RepresentabilityError, match="not representable"):
            analysis.body_acceleration_load((tiny, 0., 0.))
        assert analysis._handle.solve_count == 0
        np.testing.assert_array_equal(analysis.body_acceleration_load((0., 0., 0.)).load, 0.)
        body = analysis.body_acceleration_load((1e-200, 0., 0.))
        assert np.any(body.load != 0.)
        assert body.load[::6].sum() == pytest.approx(analysis._body_mass*1e-200, rel=2e-12, abs=0.)
        subnormal = analysis.body_acceleration_load((1e-320, 0., 0.))
        assert subnormal.load[::6].sum() == analysis._body_mass*1e-320
    finally:
        analysis.close()


def test_coupled_generalized_section_preserves_physical_body_work():
    definition = _loaded_definition("mixed", distorted=True)
    # Strict SPD with axial, shear, torsion and bending couplings; the physical
    # station trace is section-dependent even though the rigid-body mass is not.
    scales = np.diag((15., 8., 9., 2., 3., 4.))
    mixing = np.eye(6)
    mixing[0, 4], mixing[1, 5], mixing[2, 3], mixing[4, 5] = .2, -.3, .15, .25
    section = api.LinearSection.generalized(mixing.T@scales@mixing)
    definition = replace(definition, members=tuple(replace(m, section=section) for m in definition.members))
    acceleration = np.array((.3, -.8, .7))
    expected, force, moment, xyz = _oracle(definition, acceleration)
    analysis = api.create_analysis(definition=definition)
    try:
        actual = analysis.body_acceleration_load(acceleration).load
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
        poses = actual.reshape(-1, 6)
        np.testing.assert_allclose(poses[:, :3].sum(axis=0), force, rtol=3e-12, atol=3e-14)
        np.testing.assert_allclose(np.cross(xyz, poses[:, :3]).sum(axis=0)+poses[:, 3:].sum(axis=0), moment,
                                   rtol=3e-12, atol=3e-14)
        virtual = np.random.default_rng(49).normal(size=actual.shape)
        assert virtual@actual == pytest.approx(virtual@expected, rel=3e-12, abs=3e-14)
    finally:
        analysis.close()


@pytest.mark.parametrize("count", (1, 10, 100))
def test_body_load_batches_keep_factor_and_complete_cancelled_prefix(count):
    definition = _loaded_definition("mixed")
    analysis = api.create_analysis(definition=definition)
    try:
        cases = tuple((str(i), analysis.body_acceleration_load((0., 0., -(1.+i/count))).load) for i in range(count))
        batch = analysis.solve_linear_many(cases, selected_dofs=(49, 50), recover_members={})
        assert batch.status == "completed", batch.failure_reason
        assert len(batch.cases) == count and batch.factorization_count == 1
        assert batch.solve_count == count
        assert batch.member_resultant_envelopes == ()
        token = CancellationToken()
        def stop(name): token.cancel("complete first body case")
        cancelled = analysis.solve_linear_many((("first", cases[0][1]), ("never", cases[0][1])), selected_dofs=(49, 50),
            recover_members={}, cancellation_token=token, on_case=stop)
        assert cancelled.status == "cancelled" and len(cancelled.cases) == 1
        assert analysis._handle.solve_count == count+1
    finally:
        analysis.close()


@pytest.mark.parametrize("bad", ((True,0.,0.), ("1",0.,0.), (1j,0.,0.), (0.,np.nan,0.), (0.,np.inf,0.), (1.,2.)))
def test_invalid_acceleration_is_not_a_successful_load(bad):
    analysis = api.create_analysis(definition=_loaded_definition("mixed"))
    try:
        with pytest.raises(ValueError, match="acceleration"):
            analysis.body_acceleration_load(bad)
        assert analysis._handle.solve_count == 0
    finally:
        analysis.close()


def test_missing_density_mass_and_closed_owner_fail_explicitly():
    definition = _loaded_definition("s3")
    missing = (replace(definition, shells=(replace(definition.shells[0], density_kg_m3=None),)+definition.shells[1:]),
               replace(definition, members=(replace(definition.members[0], mass=None),)+definition.members[1:]))
    for model in missing:
        analysis = api.create_analysis(definition=model)
        assert "body_acceleration_load" not in analysis.capabilities.operations
        with pytest.raises(ValueError, match="density/mass"):
            analysis.body_acceleration_load((0.,0.,-1.))
        analysis.close()
        with pytest.raises(ExperimentalB2Q4PanelError, match="closed"):
            analysis.body_acceleration_load((0.,0.,-1.))


@pytest.mark.parametrize("density", (0., -1., np.nan, np.inf, True, 10**400))
def test_present_invalid_s3_density_cannot_hide_in_static_scope(density):
    definition = _loaded_definition("s3")
    definition = replace(definition, shells=(replace(definition.shells[0], density_kg_m3=density),)+definition.shells[1:])
    with pytest.raises(ValueError, match="density"):
        api.create_analysis(definition=definition)
