"""Actual coupled owner/current-rest modes; full singular saddle oracle."""
from hashlib import sha256
import numpy as np
import pytest
from scipy.linalg import eigvals
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_coupled_modes import capture, modes, buckling
from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis
from anyfesolver._ge_beam3_shell_joint_state import decode
from anyfesolver._ge_beam3_variational_shell import response as shell_response
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_coupled_free_beam import build
from test_ge_beam3_native_generalized_modal import inertia


def fixture(topology, curved=True, transform=None):
    old = build(topology, curved, transform); a = old.assembly
    successor = ShellBeamTrialAssembly(a.beam, topology=topology, coordinates=a.coordinates,
        reference_normal=a.normal, thickness=.2, elastic_modulus=1000., poisson_ratio=.3,
        shell_node=a.shell_node, beam_node=a.beam_node, variational_shell=True)
    return CoupledShellBeamAnalysis(successor, targets=old.targets, shell_fixed=old.shell_fixed,
        nodal_forces=old.forces)


def arguments(owner):
    return dict(section_inertias={eid: inertia() for eid, _ in owner.assembly.beam.elements}, shell_density=2.)


def saddle_oracle(owner, state, packet):
    response, _, _, _ = owner._evaluate(state, state.mechanical, state.shell_u, state.multipliers, state.cursor)
    inverse = np.eye(owner.assembly.count)
    for i, row in enumerate(state.shell_u.reshape(-1, 6)):
        inverse[6*i+3:6*i+6, 6*i+3:6*i+6] = np.linalg.inv(_exp_terms(row[3:])[1])
    h = response.tangent@inverse
    m = np.zeros_like(h); ids = packet.full_kinematic_dofs
    m[np.ix_(ids, ids)] = packet.full_mass
    free = owner.free
    values = eigvals(h[np.ix_(free, free)], m[np.ix_(free, free)])
    finite = values[np.isfinite(values)]
    assert np.max(np.abs(finite.imag)) < 1e-11*max(1., np.max(np.abs(finite.real)))
    return np.sort(finite.real)


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
@pytest.mark.parametrize('loaded', (False, True))
def test_actual_coupled_modes_and_full_singular_saddle(topology, curved, loaded, tmp_path):
    owner = fixture(topology, curved); result = owner.solve(stop_after=2 if loaded else 0)
    assert result.status == 'paused', result.failure
    state = result.state; before = canonical(state.descriptor()); checkpoint = result.checkpoint
    packet, actual = modes(owner, state, **arguments(owner), bounds=(-100., 10000.), num_modes=3)
    reference = saddle_oracle(owner, state, packet)
    error = np.max(np.abs(actual.eigenvalues-reference[:3])/np.maximum(1., np.abs(reference[:3])))
    assert error < 1e-11, (actual.eigenvalues, reference[:3], error)
    assert packet.internal_cell_coordinates == (12 if curved else 6)
    assert packet.shell_mass_rank == (24 if topology == 'Q4' else 9)
    assert np.count_nonzero(packet.kinetic[:, packet.algebraic_dofs]) == 0
    assert np.linalg.norm(packet.constraints@packet.lift) < 1e-11
    k = (packet.left@packet.right).T@(packet.left@packet.right)+packet.geometric
    m = packet.kinetic.T@packet.kinetic
    v = actual.high_modes+actual.low_modes
    assert np.linalg.norm(v.T@m@v-np.eye(3)) < 1e-11
    assert canonical(state.descriptor()) == before and result.checkpoint == checkpoint
    assert not owner._lock.locked() and not owner.assembly._lock.locked()
    assert packet.production_qualified is False and packet.buckling_factor_authorized is False
    (tmp_path/'modes.json').write_bytes(canonical(dict(values=actual.eigenvalues, oracle=reference,
        oracle_error=float(error), packet=packet, modes=actual, checkpoint_sha256=sha256(checkpoint).hexdigest(),
        production_qualified=False)))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_inertia_changes_roots_not_static_state_and_global_covariance(topology, tmp_path):
    owner = fixture(topology); solved = owner.solve(stop_after=2)
    assert solved.status == 'paused', solved.failure
    kwargs = arguments(owner)
    p, base = modes(owner, solved.state, **kwargs, bounds=(-100., 10000.), num_modes=3)
    scaled = dict(section_inertias={i: 4*m for i, m in kwargs['section_inertias'].items()}, shell_density=8.)
    p2, heavier = modes(owner, solved.state, **scaled, bounds=(-100., 10000.), num_modes=3)
    assert np.max(np.abs(heavier.eigenvalues*4-base.eigenvalues)) < 1e-11
    assert p.state_sha256 == p2.state_sha256 and p.inertia_sha256 != p2.inertia_sha256
    s = Rotation.from_rotvec([1.3, -.7, 2.1]).as_matrix()
    other = fixture(topology, transform=s); rotated = other.solve(stop_after=2)
    assert rotated.status == 'paused', rotated.failure
    p3, modes3 = modes(other, rotated.state, **arguments(other), bounds=(-100., 10000.), num_modes=3)
    assert np.max(np.abs(modes3.eigenvalues-base.eigenvalues)) < 1e-11
    (tmp_path/'mass-covariance.json').write_bytes(canonical(dict(values=base.eigenvalues,
        mass_scaled=heavier.eigenvalues, rotated=modes3.eigenvalues, production_qualified=False)))


@pytest.mark.parametrize('kind', ('v1', 'foreign', 'zero-density', 'missing-inertia', 'nonfinite-inertia', 'cancel'))
def test_spectral_admission_and_atomic_failure(kind):
    owner = fixture('Q4'); result = owner.solve(stop_after=0); kwargs = arguments(owner)
    state = result.state
    expected = ValueError
    if kind == 'v1':
        owner = build('Q4', True); result = owner.solve(stop_after=0); state = result.state
    elif kind == 'foreign': state = fixture('Q4').solve(stop_after=0).state
    elif kind == 'zero-density': kwargs['shell_density'] = 0.
    elif kind == 'missing-inertia': kwargs['section_inertias'].clear()
    elif kind == 'nonfinite-inertia': kwargs['section_inertias'][1][0, 0] = np.nan
    else:
        token = CancellationToken(); token.cancel(); kwargs['cancellation_token'] = token; expected = SolveCancelled
    with pytest.raises(expected): capture(owner, state, **kwargs)
    assert not owner._lock.locked() and not owner.assembly._lock.locked()


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_buckling_without_mass_and_uncondensed_multiplier_pencil(topology, curved, tmp_path):
    base = fixture(topology, curved); a = base.assembly
    force = np.zeros_like(base.forces); force[-6] = -.02
    owner = CoupledShellBeamAnalysis(a, targets=(.5, 1.), shell_fixed=base.shell_fixed, nodal_forces=force)
    run = owner.solve()
    assert run.status == 'completed', run.failure
    before = canonical(run.state.descriptor())
    packet, actual = buckling(owner, run.state, bounds=(0., 1000.), num_modes=3)
    assert actual.status == 'COMPLETED' and len(actual.multipliers) == 3
    assert packet.mass_captured is False and packet.full_mass is None and packet.kinetic.shape[0] == 0
    assert a.model.get_material('joint-shell').density == 0.
    # Independent generalized eigenproblem retains original stress unknowns,
    # all cell rotations and the six joint multipliers. No static lambda=1 lift.
    response_, _, _, _ = owner._evaluate(run.state, run.state.mechanical, run.state.shell_u,
        run.state.multipliers, run.state.cursor)
    inverse = np.eye(a.count)
    for i, row in enumerate(run.state.shell_u.reshape(-1, 6)):
        inverse[6*i+3:6*i+6, 6*i+3:6*i+6] = np.linalg.inv(_exp_terms(row[3:])[1])
    raw = response_.tangent@inverse
    _, _, _, smat, _ = shell_response(a.model, a.element, run.state.shell_u,
        decode(run.state.shell_history), 3, split=True)
    baseline = raw.copy(); kinetic_ids = packet.full_kinematic_dofs
    baseline[np.ix_(kinetic_ids, kinetic_ids)] = 0.
    baseline[:a.shell_count, :a.shell_count] = smat
    stress = raw-baseline
    # Invert the nonsingular zero-load material saddle, not the singular
    # stress matrix. Its algebraic infinity branches map to mu=0 rather than
    # roundoff-sized complex reciprocals around 1e13. The unchanged requested
    # lambda window (0,1000] corresponds to mu >= 1/1000.
    inverse_roots = eigvals(np.linalg.solve(baseline[np.ix_(owner.free, owner.free)],
        -stress[np.ix_(owner.free, owner.free)]))
    selected = inverse_roots[inverse_roots.real >= 1/1000.]
    assert np.max(np.abs(selected.imag)/np.maximum(1., np.abs(selected.real))) < 1e-11
    expected = np.sort(1/selected.real)[:3]
    error = float(np.max(np.abs(actual.multipliers-expected)/np.maximum(1., np.abs(expected))))
    assert error < 1e-11, (actual.multipliers, expected, error)
    assert canonical(run.state.descriptor()) == before
    (tmp_path/'buckling.json').write_bytes(canonical(dict(packet=packet, actual=actual,
        saddle_oracle=expected, inverse_root_real=inverse_roots.real, inverse_root_imag=inverse_roots.imag,
        oracle_error=error, checkpoint_sha256=sha256(run.checkpoint).hexdigest(),
        production_qualified=False)))


def test_nonconservative_programme_rejected_before_spectrum():
    from anyfesolver._ge_beam3_retained_generalized_state import Program
    from anyfesolver._ge_beam3_coupled_beam_subdomain import CoupledBeamSubdomain
    from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
    from anyfesolver._ge_beam3_native_line_loading import LinePattern
    base = fixture('Q4'); a = base.assembly
    program = Program((1.,), DistributedPattern(LinePattern(()), ((1, .1, .0, .0), (2, .1, .0, .0))))
    beam = CoupledBeamSubdomain(a.beam.model, program)
    other = ShellBeamTrialAssembly(beam, topology='Q4', coordinates=a.coordinates,
        reference_normal=a.normal, thickness=.2, elastic_modulus=1000., poisson_ratio=.3,
        shell_node=a.shell_node, beam_node=a.beam_node, variational_shell=True)
    owner = CoupledShellBeamAnalysis(other, targets=(1.,), shell_fixed=base.shell_fixed, nodal_forces=base.forces)
    accepted = owner.solve(stop_after=0)
    with pytest.raises(ValueError, match='nonconservative'):
        capture(owner, accepted.state, **arguments(owner))
    with pytest.raises(ValueError, match='nonconservative'):
        buckling(owner, accepted.state, bounds=(0., 1000.))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_actual_active_plastic_coupled_state_is_not_elastic_spectral_authority(topology, tmp_path):
    from anyfesolver._ge_beam3_retained_generalized_state import Context, Program
    from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
    from anyfesolver._ge_beam3_native_line_loading import LinePattern
    from test_ge_beam3_native_generalized_restart import make as plastic_model
    model = plastic_model('curved-plastic')
    pattern = DistributedPattern(LinePattern(((1, .09, -.03, .02),)), ())
    beam = Context(model, Program((1.,), pattern))
    xyz = np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2], [0., 1., .2]])
    if topology == 'S3-V2D': xyz = xyz[:3]
    a = ShellBeamTrialAssembly(beam, topology=topology, coordinates=xyz,
        reference_normal=np.array([0., 0., 1.]), thickness=.2, elastic_modulus=1000.,
        poisson_ratio=.3, shell_node=2, beam_node=3, variational_shell=True)
    fixed = tuple(range(6)) if topology == 'S3-V2D' else (*range(6), *range(18, 24))
    owner = CoupledShellBeamAnalysis(a, targets=(.25, .5, 1.), shell_fixed=fixed,
        nodal_forces=np.zeros(a.shell_count+beam.nodal_count))
    run = owner.solve()
    assert run.status == 'completed', run.failure
    assert any(sum(s.accumulated) > 0. for cell in run.state.beam_histories for s in cell.stations)
    before = canonical(owner.recover(run.state))
    with pytest.raises(ValueError, match='yield|elastic-interior|material history'):
        capture(owner, run.state, **arguments(owner))
    with pytest.raises(ValueError, match='yield|elastic-interior|material history'):
        buckling(owner, run.state, bounds=(0., 1000.))
    assert canonical(owner.recover(run.state)) == before
    assert not owner._lock.locked() and not a._lock.locked()
    (tmp_path/'active-plastic.json').write_bytes(canonical(dict(checkpoint=run.checkpoint.decode('ascii'),
        recovery=before.decode('ascii'), spectral_rejected=True, production_qualified=False)))


@pytest.mark.parametrize('offset', (0., 1e-8, 1e-12))
def test_exact_zero_versus_small_real_eccentric_inertia(offset, tmp_path):
    base = fixture('S3-V2D', False); a = base.assembly
    xyz = a.coordinates-a.coordinates[a.shell_node]+a.beam.reference_positions[0]
    xyz[:, 2] += offset
    other = ShellBeamTrialAssembly(a.beam, topology='S3-V2D', coordinates=xyz,
        reference_normal=a.normal, thickness=.2, elastic_modulus=1000., poisson_ratio=.3,
        shell_node=a.shell_node, beam_node=a.beam_node, variational_shell=True)
    owner = CoupledShellBeamAnalysis(other, targets=(1.,), shell_fixed=base.shell_fixed, nodal_forces=base.forces)
    state = owner.solve(stop_after=0).state
    packet = capture(owner, state, **arguments(owner))
    # Six unattached beam trace rotations, three other free shell rotations,
    # and three/one shared attachment spin directions are exactly massless.
    assert len(packet.algebraic_dofs) == (12 if offset == 0. else 10)
    dynamic = [i for i in range(packet.lift.shape[1]) if i not in packet.algebraic_dofs]
    assert np.all(np.max(np.abs(packet.kinetic[:, dynamic]), axis=0) > 0.)
    assert np.count_nonzero(packet.kinetic[:, packet.algebraic_dofs]) == 0
    (tmp_path/'small-offset.json').write_bytes(canonical(dict(offset=offset, kinetic=packet.kinetic,
        lift=packet.lift, algebraic=packet.algebraic_dofs, production_qualified=False)))


def test_inertia_mutation_during_spectral_solution_rejected(monkeypatch):
    from anyfesolver import _ge_beam3_coupled_modes as module
    owner = fixture('Q4', False); state = owner.solve(stop_after=0).state; kwargs = arguments(owner)
    def mutate(*args, **controls):
        kwargs['section_inertias'][1][0, 0] += 1.
        return object()  # Injected failed consumer, never a scientific result.
    monkeypatch.setattr(module, 'solve_paired_factor_chain_modes', mutate)
    with pytest.raises(ValueError, match='inertia changed'):
        modes(owner, state, **kwargs, bounds=(-100., 10000.), num_modes=3)
    assert not owner._lock.locked() and not owner.assembly._lock.locked()
