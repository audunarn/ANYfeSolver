"""Native lumped-S3 inertia and bounded descriptor controls, not continuum qualification."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.spatial.transform import Rotation

from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver._stiffened_shell_physical_q4_mass import POLICY as Q4_MASS_POLICY
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as owner_module
from test_stiffened_shell_b2_mixed_panel_experimental import (
    _definition as _static_definition, _dense_operators,
)


def _definition(kind="mixed", *, free_interior_spin=False):
    original, free = _static_definition(kind)
    if free_interior_spin:
        free = np.union1d(free, (11, 17, 29, 35))
    shells = tuple(replace(shell, density_kg_m3=2.) for shell in original.shells)
    mass = api.MassMetadata("declared-centered-inertia", (.02, -.03, .01),
                           np.diag((.3, .3, .3, .003, .001, .002)))
    members = tuple(replace(member, mass=mass) for member in original.members)
    return api.B2MixedReferenceModalDefinition(**{
        **original.__dict__, "shells": shells, "members": members,
        "transformation": sparse.eye(54, format="csr")[:, free]}), free


def _analytic_facet_mass(shell, nodes):
    vertices = np.array([nodes[node] for node in shell.node_ids])
    normal = np.asarray(shell.reference_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    section = np.zeros((6, 6))
    section[:3, :3] = shell.density_kg_m3*shell.thickness_m*np.eye(3)
    if type(shell) is api.S3V2DPanelShell:
        area = np.linalg.norm(np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]))/2.
        return np.kron(area/3.*np.eye(3), section)
    # These fixtures are rectangles, including rigidly rotated rectangles.
    area = np.linalg.norm(np.cross(vertices[1]-vertices[0], vertices[3]-vertices[0]))
    nodal = area/36.*np.array(((4., 2., 1., 2.), (2., 4., 2., 1.),
                              (1., 2., 4., 2.), (2., 1., 2., 4.)))
    section[3:, 3:] = shell.density_kg_m3*shell.thickness_m**3/12.*(
        np.eye(3)-np.outer(normal, normal))
    return np.kron(nodal, section)


def _dense_mass(definition):
    """Analytic facet inertia and independent six-point station kinetic integration."""
    mass = np.zeros((54, 54))
    shell_index = {node: i for i, node in enumerate(sorted(definition.shell_nodes_m))}
    beam_index = {node: i+6 for i, node in enumerate((10, 11, 12))}
    for shell in definition.shells:
        slots = [6*shell_index[node]+j for node in shell.node_ids for j in range(6)]
        mass[np.ix_(slots, slots)] += _analytic_facet_mass(shell, definition.shell_nodes_m)
    abscissae, weights = np.polynomial.legendre.leggauss(6)
    stations = tuple(float((x+1.)/2.) for x in abscissae)
    for member in definition.members:
        jacobians = member.trace_global(np.zeros(12), stations=stations).jacobian
        x, y, z = member.mass.origin_local_m
        origin = np.eye(6)
        origin[:3, 3:] = ((0., z, -y), (-z, 0., x), (y, -x, 0.))
        local_to_global = linalg.block_diag(member.local_frame_rows().T,
                                            member.local_frame_rows().T)
        kinetic = origin@local_to_global.T
        inertia = kinetic.T@np.array(member.mass.matrix_per_m)@kinetic
        local = sum(member.length_m*weight/2.*(jacobian.T@inertia@jacobian)
                    for weight, jacobian in zip(weights, jacobians))
        slots = [6*beam_index[node]+j for node in member.node_ids for j in range(6)]
        mass[np.ix_(slots, slots)] += local
    return mass


@pytest.mark.parametrize("reverse", (False, True))
def test_native_triangle_mass_matches_analytic_lumps_and_nine_zero_rotations(reverse):
    rotation = Rotation.from_rotvec((.2, -.4, .3)).as_matrix()
    xyz = np.array(((0., 0., 0.), (1.2, 0., 0.), (.2, .8, 0.)))@rotation.T
    nodes = {i+1: tuple(point) for i, point in enumerate(xyz)}
    normal = tuple(rotation[:, 2])
    ids = (1, 3, 2) if reverse else (1, 2, 3)
    shell = api.S3V2DPanelShell("triangle", ids, 1000., .3, .15, normal, 2.)
    model = FEModel("independent-native-mass")
    model.add_material("shell", shell.E, shell.nu, density=shell.density_kg_m3)
    for node, point in nodes.items(): model.add_node(node, *point)
    element = create_shell_element(1, list(ids), "shell", formulation="e4-pl-s3-v2d",
        thickness=shell.thickness_m, reference_normal=normal)
    model.add_element(1, element)
    mass = element.compute_mass_matrix(model.mesh, model.get_material("shell"))
    expected = _analytic_facet_mass(shell, nodes)
    np.testing.assert_allclose(mass, expected, rtol=2e-14, atol=2e-16)
    angular = [6*i+j for i in range(3) for j in (3, 4, 5)]
    np.testing.assert_array_equal(mass[angular], 0.)
    np.testing.assert_array_equal(mass[:, angular], 0.)
    assert np.linalg.matrix_rank(mass) == 9
    velocity = np.tile((.7, -.2, .4, .1, .8, -.6), 3)
    total_mass = 2.*.15*.48
    np.testing.assert_allclose(velocity@mass@velocity, total_mass*np.dot(velocity[:3], velocity[:3]), rtol=2e-14)
    np.testing.assert_allclose(np.diag(mass)[::6]@xyz, total_mass*xyz.mean(axis=0), atol=2e-15)


@pytest.mark.parametrize("kind", ("s3", "mixed"))
def test_assembled_mass_reversal_and_global_frame_covariance(kind):
    definition, _ = _definition(kind)
    reverse = replace(definition, shells=tuple(
        replace(shell, node_ids=(shell.node_ids[0], shell.node_ids[2], shell.node_ids[1]))
        if type(shell) is api.S3V2DPanelShell else shell for shell in definition.shells))
    rotation = Rotation.from_rotvec((.2, -.4, .3)).as_matrix()
    moved = replace(definition,
        shell_nodes_m={node: tuple(rotation@np.array(point)+(.4, -.2, .7))
                       for node, point in definition.shell_nodes_m.items()},
        shells=tuple(replace(shell, reference_normal=tuple(rotation@np.array(shell.reference_normal)))
                     for shell in definition.shells),
        members=tuple(replace(member,
            reference_points_m=tuple(tuple(rotation@np.array(point)+(.4, -.2, .7))
                                     for point in member.reference_points_m),
            local_y_global=tuple(rotation@np.array(member.local_y_global))) for member in definition.members))
    owners = [api.create_analysis(definition=item) for item in (definition, reverse, moved)]
    try:
        mass = _dense_mass(definition)
        np.testing.assert_allclose(owners[0]._M.toarray(), mass, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(owners[1]._M.toarray(), mass, rtol=2e-13, atol=2e-15)
        transform = np.kron(np.eye(9), linalg.block_diag(rotation, rotation))
        np.testing.assert_allclose(owners[2]._M.toarray(), transform@mass@transform.T,
                                   rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(owners[2]._M.toarray(), _dense_mass(moved), rtol=2e-13, atol=2e-15)
        # This tests inertia covariance. The tilted panel's coordinate support
        # subset is not asserted to represent the original physical supports.
    finally:
        for owner in owners: owner.close()


@pytest.mark.parametrize("kind", ("s3", "mixed"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("free_interior_spin", (False, True))
def test_sparse_modes_match_independent_singular_dense_pencil(kind, reverse, free_interior_spin):
    definition, free = _definition(kind, free_interior_spin=free_interior_spin)
    if reverse:
        definition = replace(definition, shells=tuple(
            replace(shell, node_ids=(shell.node_ids[0], shell.node_ids[2], shell.node_ids[1]))
            if type(shell) is api.S3V2DPanelShell else shell for shell in definition.shells))
    K, C = _dense_operators(definition)
    M = _dense_mass(definition)
    spin = np.array((11, 17, 29, 35))
    assert bool(np.all(np.isin(spin, free))) == free_interior_spin
    assert not np.any(np.isin(np.r_[0:6, 18:24, 36:42], free))
    np.testing.assert_array_equal(M[spin], 0.)
    np.testing.assert_array_equal(M[:, spin], 0.)
    assert np.all(np.linalg.norm(K[spin], axis=1) > 0.)
    A = np.block([[K[np.ix_(free, free)], C[:, free].T],
                  [C[:, free], np.zeros((len(C), len(C)))]])
    B = linalg.block_diag(M[np.ix_(free, free)], np.zeros((len(C), len(C))))
    values, vectors = linalg.eig(A, B)
    assert np.isinf(values).any()  # Constraints and S3 angular DOFs remain algebraic.
    finite = np.flatnonzero(np.isfinite(values))
    assert np.max(abs(values[finite].imag)) <= 1e-8*np.max(abs(values[finite].real))
    assert np.min(values[finite].real) > 0.
    order = finite[np.argsort(values[finite].real)]
    expected = np.sqrt(values[order[:3]].real)/(2.*np.pi)
    owner = api.create_analysis(definition=definition)
    try:
        np.testing.assert_allclose(owner._M.toarray(), M, rtol=2e-13, atol=2e-15)
        result = owner.modes(api.B2ReferenceModesRequest(count=3))
        assert result.status == "ok", result.failure_reason
        np.testing.assert_allclose(result.frequencies_hz, expected, rtol=2e-8)
        physical = np.array(result.physical_modes).T
        np.testing.assert_allclose(physical.T@M@physical, np.eye(3), atol=1e-7)
        np.testing.assert_array_equal(physical[np.setdiff1d(np.arange(54), free)], 0.)
        for frequency, u, lam in zip(result.frequencies_hz, physical.T, result.point_multiplier_modes):
            balance = K@u+C.T@lam-(2.*np.pi*frequency)**2*(M@u)
            bound = np.linalg.norm(K@u)+np.linalg.norm(C.T@lam)+np.linalg.norm((2.*np.pi*frequency)**2*M@u)
            assert np.linalg.norm(balance[free]) <= 1e-8*bound
            if free_interior_spin:
                assert np.max(abs(balance[spin])) <= 1e-8*bound
            assert np.linalg.norm(C@u) <= 1e-9*np.linalg.norm(abs(C)@abs(u))
        # Compare the first three physical eigenspaces in the kinetic metric;
        # signs and degenerate-basis choices are not separate observables.
        dense = np.zeros((54, 3))
        dense[free] = vectors[:len(free), order[:3]].real
        d, basis = linalg.eigh(dense.T@M@dense)
        dense = dense@basis@np.diag(1./np.sqrt(d))
        np.testing.assert_allclose(linalg.svdvals(dense.T@M@physical), np.ones(3), atol=2e-7)
        caps = owner.capabilities
        assert isinstance(caps, api.ExperimentalB2MixedReferenceModalCapabilities)
        assert caps.inertia_policy == api.S3_V2D_TRANSLATIONAL_LUMPED_SECTION_AREAL_MASS_V1
        assert caps.shell_mass_policies == tuple((shell.shell_id,
            caps.inertia_policy if type(shell) is api.S3V2DPanelShell else Q4_MASS_POLICY)
            for shell in definition.shells)
        assert "modes" in caps.operations and not caps.production_qualified
        assert "omitted" in caps.s3_rotary_inertia and "heterogeneous" in caps.spectral_domain
        assert result.factorization_count == 1 and result.solve_count > 0
    finally:
        owner.close()


@pytest.mark.parametrize("density", (None, 0., -1., True, float("nan"), float("inf"),
                                      np.nextafter(0., 1.)))
def test_incomplete_invalid_or_underflow_s3_density_is_rejected(density):
    definition, _ = _definition("s3")
    shells = (replace(definition.shells[0], density_kg_m3=density),)+definition.shells[1:]
    with pytest.raises(ValueError, match="density|mass_per_area|representable"):
        api.create_analysis(definition=replace(definition, shells=shells))


@pytest.mark.parametrize("bad", ("missing-beam-mass", "offset", "mpc", "free-free", "footprint", "policy", "no-policy", "geometric"))
def test_modal_definition_rejects_unsupported_inputs(bad):
    definition, _ = _definition()
    if bad == "missing-beam-mass":
        definition = replace(definition, members=(replace(definition.members[0], mass=None),)+definition.members[1:])
    elif bad == "offset":
        offset = definition.prescribed_offset.copy(); offset[0] = 1e-6
        definition = replace(definition, prescribed_offset=offset)
    elif bad == "mpc":
        T = definition.transformation.tolil(); T[1, 0] = .2
        definition = replace(definition, transformation=T.tocsr())
    elif bad == "free-free": definition = replace(definition, transformation=sparse.eye(54, format="csr"))
    elif bad == "footprint":
        definition = replace(definition, footprints=(api.B2NodalRigidFootprint("patch", (2, 3, 6), 12),))
    elif bad == "policy": definition = replace(definition, inertia_policy="invented")
    elif bad == "no-policy": definition = replace(definition, inertia_policy=None)
    elif bad == "geometric":
        from anyfesolver._b2_physical_linear_definition import GeometricMetadata
        geometry = GeometricMetadata("geometry", (0., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        definition = replace(definition, members=(replace(definition.members[0], geometric=geometry),)+definition.members[1:])
    with pytest.raises(ValueError): api.create_analysis(definition=definition)


def test_opt_in_policy_is_frozen_and_static_and_legacy_routes_unchanged():
    assert api.Q4_DIRECTOR_MASS_POLICY == Q4_MASS_POLICY
    definition, _ = _definition()
    with pytest.raises(FrozenInstanceError): definition.inertia_policy = "changed"
    values = {name: value for name, value in definition.__dict__.items()
              if name not in ("inertia_policy", "boundary_policy")}
    static = api.create_analysis(definition=api.B2MixedPanelDefinition(**values))
    q4, _ = _definition("q4")
    q4_values = {name: value for name, value in q4.__dict__.items()
                 if name not in ("inertia_policy", "boundary_policy")}
    legacy = api.create_analysis(definition=api.B2Q4PanelDefinition(**q4_values))
    try:
        assert static._M is None and "modes" not in static.capabilities.operations
        with pytest.raises(ValueError, match="native S3 V2D.*deferred"):
            static.modes(api.B2ReferenceModesRequest())
        assert type(legacy.capabilities) is owner_module.ExperimentalB2Q4PanelCapabilities
        assert "modes" in legacy.capabilities.operations
        assert legacy.capabilities.point_law_id == owner_module.LAW_ID
    finally:
        static.close(); legacy.close()


@pytest.mark.parametrize("failure", ("cancel-before", "cancel-action", "arpack", "backend"))
def test_modal_cancellation_and_numerical_failure_publish_no_partial_pairs(failure, monkeypatch):
    definition, _ = _definition("s3")
    owner = api.create_analysis(definition=definition)
    token = CancellationToken()
    if failure == "cancel-before": token.cancel("before reference modes")
    elif failure == "cancel-action":
        def interrupt(operator, **kwargs):
            token.cancel("during reference modes")
            operator@np.ones(operator.shape[0])
            raise AssertionError("cancelled action returned")
        monkeypatch.setattr(owner_module, "eigs", interrupt)
    elif failure == "arpack":
        def partial(*args, **kwargs):
            raise ArpackNoConvergence("partial mixed eigensolve", np.array((1.,)), np.ones((44, 1)))
        monkeypatch.setattr(owner_module, "eigs", partial)
    elif failure == "backend":
        original = owner_module.factorize_cached
        def broken(*args, **kwargs):
            handle = original(*args, **kwargs)
            def fail(_rhs): raise RuntimeError("Sparse solve produced NaN/Inf values")
            monkeypatch.setattr(handle, "solve", fail)
            return handle
        monkeypatch.setattr(owner_module, "factorize_cached", broken)
    try:
        result = owner.modes(api.B2ReferenceModesRequest(), cancellation=token)
        assert result.status == ("cancelled" if failure.startswith("cancel") else "failed")
        assert result.physical_modes == result.frequencies_hz == () and result.failure_reason
        assert result.factorization_count == (0 if failure == "cancel-before" else 1)
        monkeypatch.undo()
        assert owner.solve_linear(np.zeros(54)).status == "completed"
    finally:
        owner.close()


def test_unclassified_modal_backend_runtime_error_propagates(monkeypatch):
    definition, _ = _definition("s3")
    owner = api.create_analysis(definition=definition)
    original = owner_module.factorize_cached
    error = RuntimeError("unclassified mixed modal programming failure")
    try:
        with monkeypatch.context() as guard:
            def broken(*args, **kwargs):
                handle = original(*args, **kwargs)
                def fail(_rhs): raise error
                guard.setattr(handle, "solve", fail)
                return handle
            guard.setattr(owner_module, "factorize_cached", broken)
            with pytest.raises(RuntimeError, match="unclassified mixed modal programming failure") as caught:
                owner.modes(api.B2ReferenceModesRequest())
            assert caught.value is error
        assert owner.solve_linear(np.zeros(54)).status == "completed"
    finally:
        owner.close()
