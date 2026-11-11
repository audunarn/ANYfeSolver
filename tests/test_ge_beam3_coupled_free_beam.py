"""Beam support comes from the real shell/joint, never a hidden beam clamp."""
from hashlib import sha256
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_coupled_beam_subdomain import CoupledBeamSubdomain
from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly
from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis
from anyfesolver._ge_beam3_retained_generalized_state import Context, Program
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_generalized_modal import make


def build(topology='Q4', curved=False, transform=None):
    s = np.eye(3) if transform is None else transform
    model, _, _ = make(curved, 2 if curved else 1, clamped=False, transform=s)
    p = Program((1.,), DistributedPattern(LinePattern(()), ()))
    beam = CoupledBeamSubdomain(model, p)
    xyz = np.array([[0., 0., .2], [1., 0., .2], [1., 1., .2], [0., 1., .2]])
    if topology == 'S3-V2D': xyz = xyz[:3]
    a = ShellBeamTrialAssembly(beam, topology=topology, coordinates=xyz@s.T,
        reference_normal=s@np.array([0., 0., 1.]), thickness=.2, elastic_modulus=1000.,
        poisson_ratio=.3, shell_node=2, beam_node=1)
    forces = np.zeros(a.shell_count+beam.nodal_count)
    forces[-6:-3] = s@np.array([0., .0001, .0002])
    fixed = tuple(range(6)) if topology == 'S3-V2D' else (*range(6), *range(18, 24))
    return CoupledShellBeamAnalysis(a, targets=(.5, 1., 0.), shell_fixed=fixed, nodal_forces=forces)


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
@pytest.mark.parametrize('curved', (False, True))
def test_joint_is_the_only_beam_support_and_replay_is_exact(topology, curved, tmp_path):
    made = build(topology, curved); beam = made.assembly.beam
    assert beam.fixed == () and not beam.model.boundary_conditions
    assert beam.initial.accepted_state is False
    assert not hasattr(beam, 'checkpoint') and not hasattr(beam, 'record')
    with pytest.raises(ValueError, match='supported'):
        Context(beam.model, beam.program)
    with pytest.raises(ValueError, match='global coupled owner'):
        made.assembly.evaluate(beam.initial, beam.initial.mechanical,
            np.zeros(made.assembly.shell_count), np.zeros(6), parameter=0.)
    first = made.solve(stop_after=2)
    assert first.status == 'paused', first.failure
    assert np.linalg.norm(first.state.multipliers) > 1e-8
    assert np.linalg.norm(first.state.mechanical.positions-beam.reference_positions) > 1e-7
    second = build(topology, curved)
    resumed = second.solve(checkpoint=first.checkpoint, expected_sha256=sha256(first.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    full = build(topology, curved).solve()
    assert full.status == 'completed', full.failure
    assert resumed.checkpoint == full.checkpoint
    assert np.linalg.norm(resumed.state.mechanical.positions-beam.reference_positions) < 1e-10
    (tmp_path/'supported-through-joint.json').write_bytes(resumed.checkpoint)


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_large_common_reference_rotation_preserves_coupled_solution(topology, tmp_path):
    reference = build(topology, True); expected = reference.solve(stop_after=2)
    assert expected.status == 'paused', expected.failure
    s = Rotation.from_rotvec(np.array([1.3, -.7, 2.1])).as_matrix()
    transformed = build(topology, True, s); actual = transformed.solve(stop_after=2)
    assert actual.status == 'paused', actual.failure
    errors = dict(
        positions=float(np.linalg.norm(actual.state.mechanical.positions-expected.state.mechanical.positions@s.T)),
        frames=float(np.linalg.norm(actual.state.mechanical.nodal_frames-s@expected.state.mechanical.nodal_frames)),
        shell=float(np.linalg.norm(actual.state.shell_u.reshape(-1, 3)-expected.state.shell_u.reshape(-1, 3)@s.T)),
        material_multipliers=float(np.linalg.norm(actual.state.multipliers-expected.state.multipliers)))
    assert max(errors.values()) < 1e-11, errors
    (tmp_path/'covariance.json').write_bytes(canonical(dict(errors=errors,
        reference=expected.state.descriptor(), transformed=actual.state.descriptor(),
        proper_rotation=s, production_qualified=False)))


@pytest.mark.parametrize('kind', ('slots', 'fixed', 'program', 'frames'))
def test_subdomain_layout_mutation_rejected(kind):
    made = build(); beam = made.assembly.beam
    if kind == 'slots': beam.slots[0][0], beam.slots[0][1] = beam.slots[0][1], beam.slots[0][0]
    elif kind == 'fixed': beam.fixed = (0,)
    elif kind == 'program': object.__setattr__(beam.program, 'targets', (.5,))
    else: beam.reference_frames = beam.reference_frames.copy(); beam.reference_frames[0] *= -1
    with pytest.raises(ValueError): made.solve()
    assert not made._lock.locked() and not made.assembly._lock.locked()


def test_free_rigid_component_rejected_before_global_genesis():
    made = build(); a = made.assembly
    with pytest.raises(ValueError, match='free rigid motion'):
        CoupledShellBeamAnalysis(a, targets=(1.,), shell_fixed=(), nodal_forces=made.forces)
    assert not a._lock.locked()
