"""Development checks for complete retained multi-native/S3 work and state."""
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver.general_contact import create_native_contact_analysis
from anyfesolver.general_contact.native_s3_swept import native_s3_swept_admissible
from anyfesolver.control import CancellationToken

_path = Path(__file__).resolve().parents[1]/'examples/general_contact_native_multiple_s3_static.py'
_spec = importlib.util.spec_from_file_location('native_multiple_s3_example', _path)
_example = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_example)


def rebuild(owner, *, pairs=None, **settings):
    return create_native_contact_analysis(tuple(NativeBeamDefinition(raw) for raw in owner.native._definitions),
        tuple(owner.native.model.boundary_conditions),
        owner.program, owner.shell_model, owner.pairs if pairs is None else pairs,
        equilibrium_tolerance=1e-8, **settings)


def independent_energy(owner, mechanical, displacement):
    """Direct retained curve and exact quadratic gap roots; no contact kernel."""
    total = 0.
    for pair, i in zip(owner.pairs, owner.beam_indices):
        k, skin = pair.work.contact, pair.work.shell
        vertices = skin.reference_positions+(skin.work_map@displacement).reshape(3, 3)
        normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
        normal /= np.linalg.norm(normal)
        points = mechanical.positions[owner.context.nodes[i]]+mechanical.position_low[owner.context.nodes[i]]
        for cell in (0, 1):
            lift = k.beam_reference.half_cell_lift(cell, .5)
            moving = mechanical.cell_rotations[i, cell]@lift
            a = (points[cell]-vertices[0])@normal-k.radius
            b = (points[cell+1]-points[cell]+4*moving)@normal
            c = -4*moving@normal
            roots = np.polynomial.polynomial.polyroots((a, b, c))
            ends = sorted([0., 1.]+[float(r.real) for r in roots if abs(r.imag)<1e-12 and 0<r.real<1])
            reference_delta = k.beam_reference.coordinates[cell+1]-k.beam_reference.coordinates[cell]
            def integrand(s):
                gap = a+b*s+c*s*s
                measure = np.linalg.norm(reference_delta+4*(1-2*s)*lift)
                return .5*k.stiffness*min(gap, 0.)**2*measure
            total += sum(quad(integrand, left, right, epsabs=1e-13, epsrel=1e-11)[0]
                         for left, right in zip(ends, ends[1:]))
    return total


@pytest.mark.parametrize('beams', (1, 2))
def test_retained_internal_work_and_all_cross_blocks(beams):
    owner = _example.analysis(beams=beams)
    n = owner.context.count; u = np.zeros(len(owner.shell_stiffness))
    step = np.zeros(n)
    for i in range(beams): step[owner.context.nodal_count+24*i] = -.2
    pose = owner.context.advance(owner.initial.mechanical, step)
    np.testing.assert_array_equal(pose.positions, owner.initial.mechanical.positions)
    result = owner._contact(pose, u)
    assert result.contact.active_intervals > 0
    expected = independent_energy(owner, pose, u)
    assert abs(expected-result.contact.potential_j) < 1e-10
    direction = np.zeros(n+len(u))
    for i in range(beams):
        direction[owner.context.nodal_count+24*i] = .03
        algebraic = np.arange(owner.context.nodal_count+24*i+6, owner.context.nodal_count+24*(i+1))
        np.testing.assert_array_equal(result.residual[algebraic], 0.)
        assert result.tangent[algebraic, :].nnz == 0
    direction[n+owner.shell_free] = .002
    h = 1e-5
    plus = owner.context.advance(pose, h*direction[:n])
    minus = owner.context.advance(pose, -h*direction[:n])
    derivative = (independent_energy(owner, plus, u+h*direction[n:])-
                  independent_energy(owner, minus, u-h*direction[n:]))/(2*h)
    assert abs(derivative-result.residual@direction)/max(1., abs(derivative)) < 1e-6
    assert result.tangent[:n, n:].nnz and result.tangent[n:, :n].nnz
    np.testing.assert_allclose(result.contact.beam_force_n+result.contact.shell_force_n, 0., atol=1e-9)
    np.testing.assert_allclose(result.contact.beam_moment_nm+result.contact.shell_moment_nm, 0., atol=1e-9)


@pytest.mark.parametrize('beams', (1, 2))
def test_coupled_solve_and_spatial_tangent(beams):
    owner = _example.analysis(beams=beams)
    run = owner.solve()
    assert run.status == 'completed', run.failure
    accepted = run.accepted; n = owner.context.count
    assert np.all(np.abs(accepted.shell_displacement[owner.shell_free])>1e-10)
    residual, tangent, metrics, _, contact = owner._assemble(accepted.mechanical,
        accepted.shell_displacement, 1., accepted.native_histories)
    assert max(metrics) <= owner.equilibrium_tolerance
    assert owner._contact_admissible(contact)
    np.testing.assert_allclose(contact.contact.beam_force_n+contact.contact.shell_force_n,0.,atol=1e-9)
    np.testing.assert_allclose(contact.contact.beam_moment_nm+contact.contact.shell_moment_nm,0.,atol=1e-9)
    reaction=np.zeros(3)
    for dof in owner.context.fixed:
        _, axis=divmod(dof,6)
        if axis<3: reaction[axis]+=residual[dof]
    for dof in owner.shell_fixed:
        _, axis=divmod(dof,6)
        if axis<3: reaction[axis]+=residual[n+dof]
    np.testing.assert_allclose(reaction,(0.,.01,.01*beams),atol=1e-8)
    assert abs(independent_energy(owner, accepted.mechanical, accepted.shell_displacement)-contact.contact.potential_j)<1e-10
    direction = np.zeros(len(residual))
    direction[n+owner.shell_free] = .002
    for i in range(beams):
        direction[owner.context.nodal_count+24*i] = .003
        direction[owner.context.slots[i][14]] = -.004
    h=1e-5
    plus=owner._assemble(owner.context.advance(accepted.mechanical,h*direction[:n]),
        accepted.shell_displacement+h*direction[n:],1.,accepted.native_histories)[0]
    minus=owner._assemble(owner.context.advance(accepted.mechanical,-h*direction[:n]),
        accepted.shell_displacement-h*direction[n:],1.,accepted.native_histories)[0]
    predicted=tangent@direction
    tangent_error=np.linalg.norm((plus-minus)/(2*h)-predicted)/max(1.,np.linalg.norm(predicted))
    assert tangent_error<1e-5
    record=os.environ.get('ANYSOLVER_DEVELOPMENT_RECORD')
    if record:
        Path(record).with_name(f'native_{beams}_beams_response.json').write_text(json.dumps({
            'profile':run.profile,'status':run.status,'coupled_coordinates':len(residual),
            'metrics':metrics,'directional_tangent_error':float(tangent_error),
            'shell_displacement_m':accepted.shell_displacement.tolist(),
            'contact':contact.contact.to_dict(),'checkpoint_sha256':run.checkpoint_sha256},
            default=lambda a:a.tolist(),indent=2),encoding='utf-8')


def test_complete_split_restart_and_failure_state():
    full = _example.analysis().solve()
    prefix = _example.analysis().solve(stop_after=1)
    assert full.status == 'completed' and prefix.status == 'paused'
    resumed = _example.analysis().solve(checkpoint=prefix.checkpoint,expected_sha256=prefix.checkpoint_sha256)
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == full.checkpoint
    owner = _example.analysis(iterations=1)
    failed = owner.solve()
    assert failed.status == 'failed' and failed.accepted.completed_targets == 0
    assert sha(failed.accepted.native_histories) == sha(owner.initial.native_histories)
    np.testing.assert_array_equal(failed.accepted.shell_displacement, owner.initial.shell_displacement)


def test_consumed_map_and_resources_are_bound():
    owner = _example.analysis()
    mapping = owner.pairs[0].work.work_map.copy()
    mapping.data[0] = 2.
    owner.pairs[0].work.work_map = mapping
    with pytest.raises(ValueError): owner._guard()
    with pytest.raises(ValueError): owner._contact(owner.initial.mechanical, owner.initial.shell_displacement)
    with pytest.raises(ValueError): rebuild(owner)
    owner = _example.analysis()
    with pytest.raises(MemoryError): rebuild(owner, maximum_coupled_coordinates=10)
    with pytest.raises(ValueError): rebuild(owner, maximum_pairs=1)


def test_new_factory_lazy_public_exports():
    from anyfesolver.general_contact import NativeS3ContactPair
    assert NativeS3ContactPair is type(_example.analysis().pairs[0])


def test_second_beam_hidden_rotation_sweep_and_individual_gap_gate():
    owner = _example.analysis(beams=2)
    u = owner.initial.shell_displacement
    setup = np.zeros(owner.context.count)
    for node in owner.context.nodes[1][1:]: setup[6*node+2] = .0465
    setup[owner.context.nodal_count+24+3] = -np.pi/2-.4
    start = owner.context.advance(owner.initial.mechanical, setup)
    increment = np.zeros(owner.context.count+len(u))
    increment[owner.context.nodal_count+24+3] = .8
    end = owner.context.advance(start, increment[:owner.context.count])
    midpoint = owner.context.advance(start, .5*increment[:owner.context.count])
    mid = owner._contact(midpoint, u)
    assert not owner._contact_admissible(mid)
    checks=[]
    for p, i in zip(owner.pairs, owner.beam_indices):
        nodes=owner.context.nodes[i]
        a=p.work.evaluate(start.positions[nodes],start.position_low[nodes],start.cell_rotations[i],u).contact
        b=p.work.evaluate(end.positions[nodes],end.position_low[nodes],end.cell_rotations[i],u).contact
        assert a.minimum_gap_m >= 0. and b.minimum_gap_m >= 0.
        checks.append(native_s3_swept_admissible(owner.context,p.work,i,p.gap_tolerance_m,
            p.maximum_swept_intervals,owner._recoverable_trial_error,start,u,increment,end,u,
            a.minimum_gap_m,b.minimum_gap_m,a.minimum_rim_clearance_m,b.minimum_rim_clearance_m))
    assert checks[:2] == [True, True] and checks[2] is False
    assert not owner._swept_admissible(start,u,increment,end,u)
    pairs=tuple(replace(p,gap_tolerance_m=.1 if i==0 else .0001)
                for p,i in zip(owner.pairs,owner.beam_indices))
    asymmetric=rebuild(owner,pairs=pairs)
    actual=asymmetric._contact(midpoint,u)
    assert actual.pairs[2][1].contact.minimum_gap_m > -.1
    assert not asymmetric._contact_admissible(actual)


def test_cancel_later_pair_exception_and_checkpoint_binding(monkeypatch):
    owner=_example.analysis()
    token=CancellationToken()
    cancelled=owner.solve(cancellation_token=token,progress=lambda *_:token.cancel('after accepted first target'))
    assert cancelled.status=='cancelled' and cancelled.accepted.completed_targets==1
    owner=_example.analysis()
    later=owner.pairs[-1].work
    original=later.evaluate
    def fail(*_): raise RuntimeError('injected later S3 work failure')
    def progress(*_): monkeypatch.setattr(later,'evaluate',fail)
    failed=owner.solve(progress=progress)
    assert failed.status=='failed' and failed.accepted.completed_targets==1
    assert 'later S3' in failed.failure
    assert failed.checkpoint==cancelled.checkpoint
    assert sha(failed.accepted.native_histories)==sha(cancelled.accepted.native_histories)
    monkeypatch.setattr(later,'evaluate',original)
    resumed=owner.solve(checkpoint=failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    assert resumed.status=='completed',resumed.failure
    changed=rebuild(owner,pairs=tuple(replace(p,gap_tolerance_m=.0009) for p in owner.pairs))
    with pytest.raises(ValueError,match='definition/settings binding'):
        changed.solve(checkpoint=failed.checkpoint,expected_sha256=failed.checkpoint_sha256)
    with pytest.raises(ValueError,match='external checkpoint hash'):
        owner.solve(checkpoint=failed.checkpoint,expected_sha256='0'*64)


def test_explicit_line_enforcement_sensitivity():
    runs=[]
    for regularization in (10000.,40000.):
        owner=_example.analysis(targets=(.25,.5,.75,1.),line_stiffness_n_per_m2=regularization)
        run=owner.solve()
        assert run.status=='completed',run.failure
        assert owner._contact_admissible(run.accepted.contact)
        runs.append(run)
    assert -runs[1].accepted.contact.contact.minimum_gap_m < -runs[0].accepted.contact.contact.minimum_gap_m
    record=os.environ.get('ANYSOLVER_DEVELOPMENT_RECORD')
    if record:
        Path(record).with_name('line_enforcement_sensitivity.json').write_text(json.dumps({
            'targets':[.25,.5,.75,1.],'line_stiffness_n_per_m2':[10000.,40000.],
            'minimum_gap_m':[r.accepted.contact.contact.minimum_gap_m for r in runs],
            'shell_displacement_m':[r.accepted.shell_displacement.tolist() for r in runs]},indent=2),encoding='utf-8')


def test_runnable_two_beam_example(capsys):
    _example.main()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='completed' and result['completed_targets']==2
    assert result['shell_displacement_norm_m']>1e-10 and result['minimum_gap_m']>=-.001


def test_ordinary_static_route_keeps_new_native_owner_unloaded():
    code='''import os, sys
from pathlib import Path
import anyfesolver
assert Path(anyfesolver.__file__).resolve().is_relative_to(Path(os.environ['CONTACT_SOLVER_EXPECTED_ROOT']))
assert 'anyfesolver.general_contact' not in sys.modules
from anyfesolver.boundary import LoadCase
from anyfesolver.mesh_gen import generate_beam_mesh
model=generate_beam_mesh(1.,num_divisions=1)
load=LoadCase('axial');load.add_nodal_load(2,[10.,0.,0.,0.,0.,0.])
run=anyfesolver.solve_static_nonlinear(model,load,num_steps=1,max_iterations=5)
assert run.status=='completed',run.failure_reason
assert 'anyfesolver.general_contact.native_multi_s3_static' not in sys.modules
'''
    env=os.environ.copy()
    env.setdefault('CONTACT_SOLVER_EXPECTED_ROOT',str(Path(__file__).resolve().parents[1]/'src/anyfesolver'))
    env.setdefault('PYTHONPATH',str(Path(__file__).resolve().parents[1]/'src'))
    run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
