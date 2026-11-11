"""Small native arc transactions; not engineering stability qualification."""

import json

import numpy as np
import pytest
from scipy import sparse

import anyfesolver._ge_beam3_arc_program as driver
from anyfesolver._ge_beam3_load_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_load_state import problem
from test_ge_beam3_displacement_program import arch


def done(result,count):
    assert result.status == 'completed', result.failure
    assert result.completed_targets == count and not result.production_qualified
    assert not result.displacements.flags.writeable


def reseal(value):
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    return canonical({**body,'checkpoint_sha256':sha(body)})


def test_predictor_uses_full_border_at_a_simple_stiffness_limit():
    matrix = sparse.diags([0.,2.],format='csr'); column = np.array([-1.,0.])
    direction = driver._direction(matrix,column,np.array([0,1]),np.array([1.,0.,0.]),np.ones(3))
    np.testing.assert_array_equal(direction,[1.,0.,0.])
    with pytest.raises(RuntimeError):
        driver._direction(matrix,column,np.array([0,1]),np.array([0.,0.,1.]),np.ones(3))


def test_elastic_arc_matches_force_equilibrium_at_solved_parameter():
    result = driver.solve_arc_program(problem(),driver.ArcProgram((.05,),2.))
    done(result,1)
    fixed = solve_force_program(problem(),ForceProgram((result.parameter,)))
    done(fixed,1)
    np.testing.assert_allclose(result.displacements,fixed.displacements,atol=1e-10,rtol=0.)
    np.testing.assert_allclose(result.physical_imbalance,fixed.physical_imbalance,atol=1e-10,rtol=0.)


def test_plastic_arc_split_restart_replays_direction_and_origin():
    program = driver.ArcProgram((.2,.2,.2),2.)
    full = driver.solve_arc_program(problem(plastic=True),program); done(full,3)
    first = driver.solve_arc_program(problem(plastic=True),program,stop_after=1)
    assert first.status == 'paused'
    resumed = driver.solve_arc_program(problem(plastic=True),program,checkpoint=first.checkpoint); done(resumed,3)
    assert resumed.checkpoint == full.checkpoint
    value = json.loads(full.checkpoint)
    inner = json.loads(value['element_states'][0]['state']['payload'])['material_state']
    assert any(h['accumulated'] > 0. for h in inner['histories'])
    assert len(value['records']) == 3 and value['last_origin'] is not None


@pytest.fixture(scope='module')
def accepted():
    program = driver.ArcProgram((.02,.02),2.)
    result = driver.solve_arc_program(problem(),program); done(result,2)
    return program,result.checkpoint


@pytest.mark.parametrize('kind',['metric','direction','origin','origin_direction','reaction','step','hash','schema'])
def test_resealed_arc_restart_mutations_are_rejected(accepted,kind):
    program,raw = accepted; value = json.loads(raw)
    if kind == 'metric': value['metric'][0] *= 2.
    elif kind == 'direction':
        value['direction'] = [-x for x in value['direction']]
        value['records'][-1]['direction'] = value['direction']
    elif kind == 'origin': value['last_origin']['total'][12] += .01
    elif kind == 'origin_direction': value['last_origin']['direction'][-1] *= -1.
    elif kind == 'reaction': value['physical_imbalance'][0] += .01
    elif kind == 'step': value['records'][-1]['step_size'] = .03
    elif kind == 'hash': value['model_sha256'] = '0'*64
    elif kind == 'schema': value['schema'] = 'GE_BEAM3_TRANSLATION_CONTROL_CHECKPOINT_V1'
    with pytest.raises(ValueError): driver.solve_arc_program(problem(),program,checkpoint=reseal(value))


@pytest.mark.parametrize('stage',['native_arc.before_commit','native_arc.committed'])
def test_arc_cancellation_returns_only_last_accepted_state(stage):
    program = driver.ArcProgram((.02,.02),2.); token = CancellationToken()
    expected = 0 if stage.endswith('before_commit') else 1
    def cancel(event):
        if event['stage'] == stage: token.cancel('arc fixture')
    result = driver.solve_arc_program(problem(),program,cancellation_token=token,progress=cancel)
    assert result.status == 'cancelled' and result.completed_targets == expected
    retained = driver.solve_arc_program(problem(),program,checkpoint=result.checkpoint,stop_after=expected)
    assert retained.checkpoint == result.checkpoint


def test_arc_staging_failure_does_not_commit(monkeypatch):
    program = driver.ArcProgram((.02,),2.)
    virgin = driver.solve_arc_program(problem(),program,stop_after=0); original = driver._capsule
    def reject(*args):
        if args[7]: raise ValueError('arc staging fixture')
        return original(*args)
    monkeypatch.setattr(driver,'_capsule',reject)
    result = driver.solve_arc_program(problem(),program)
    assert result.status == 'failed' and 'staging fixture' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint


@pytest.mark.parametrize('kwargs',[
    {'steps':(0.,),'length_scale':2.}, {'steps':(.3,),'length_scale':2.},
    {'steps':(.02,),'length_scale':0.}, {'steps':(.02,),'length_scale':2.,'initial_load_sign':0.},
    {'steps':(.02,),'length_scale':2.,'parameter_scale':float('nan')},
    {'steps':(.02,),'length_scale':1e300}, {'steps':(.02,),'length_scale':1e-300},
])
def test_invalid_arc_programs_rejected(kwargs):
    with pytest.raises(ValueError): driver.ArcProgram(**kwargs)


def test_native_arc_arch_passes_onto_decreasing_load_branch(tmp_path):
    program = driver.ArcProgram((.02,)*8,2.,nodal_forces=((55,0.,-1.,0.),))
    events = []
    result = driver.solve_arc_program(arch(),program,progress=events.append)
    with (tmp_path/'diagnostic-checkpoint.json').open('xb') as stream: stream.write(result.checkpoint)
    with (tmp_path/'diagnostic-progress.json').open('xb') as stream: stream.write(canonical(events))
    with (tmp_path/'diagnostic-status.json').open('xb') as stream:
        stream.write(canonical(dict(status=result.status,failure=result.failure,completed_steps=result.completed_targets,production_qualified=False)))
    done(result,8)
    records = json.loads(result.checkpoint)['records']; loads = [r['parameter'] for r in records]
    assert any(b < a for a,b in zip(loads,loads[1:])), loads
    assert all(r['arc_residual'] <= 1e-11 for r in records)
    assert np.linalg.norm(result.physical_imbalance[6:24]) <= 1e-11
    # No spatial mode has been removed; this checks only this coarse branch.
    restored = driver.solve_arc_program(arch(),program,checkpoint=result.checkpoint)
    assert restored.checkpoint == result.checkpoint


def test_negative_initial_orientation_follows_negative_load_branch():
    result = driver.solve_arc_program(problem(),driver.ArcProgram((.02,),2.,initial_load_sign=-1.))
    done(result,1)
    assert result.parameter < 0.


def test_zero_pattern_is_rejected_before_assembly(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('must not assemble')
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',forbidden)
    with pytest.raises(ValueError,match='nonzero declared load'):
        driver.solve_arc_program(problem(line_force=(0.,0.,0.)),driver.ArcProgram((.02,),2.))


def test_arc_iteration_failure_retains_virgin_capsule_without_retry():
    program = driver.ArcProgram((.2,),2.,max_iterations=0)
    virgin = driver.solve_arc_program(problem(),program,stop_after=0)
    result = driver.solve_arc_program(problem(),program)
    assert result.status == 'failed' and 'iteration bound' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint


def test_arc_deadline_returns_only_previous_accepted_capsule(monkeypatch):
    from types import SimpleNamespace
    program = driver.ArcProgram((.02,),2.)
    virgin = driver.solve_arc_program(problem(),program,stop_after=0)
    clocks = iter((0.,601.))
    monkeypatch.setattr(driver,'time',SimpleNamespace(monotonic=lambda: next(clocks)))
    result = driver.solve_arc_program(problem(),program)
    assert result.status == 'failed' and 'deadline' in result.failure
    assert result.checkpoint == virgin.checkpoint


@pytest.mark.parametrize('kind',['duplicate','nonfinite','whitespace','missing'])
def test_invalid_arc_json_is_rejected(accepted,kind):
    program,raw = accepted
    if kind == 'duplicate': raw = raw[:-2]+b',"schema":"duplicate"}\n'
    elif kind == 'nonfinite': raw = b'{"bad":NaN}\n'
    elif kind == 'whitespace': raw += b' '
    elif kind == 'missing': raw = b'{}\n'
    with pytest.raises(ValueError): driver.solve_arc_program(problem(),program,checkpoint=raw)


def test_native_arc_is_covariant_under_a_proper_coordinate_frame_change():
    from copy import deepcopy
    from anyfesolver.fe_core import FEModel
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
    from anyfesolver._ge_beam3_p5.algebra import rotation
    original = problem(); rotated = FEModel('rotated-native-arc'); s = rotation(np.array([.7,-.4,.3]))
    for i,node in original.mesh.nodes.items(): rotated.add_node(i,*(s@np.array(node.coords())))
    for i,e in original.mesh.elements.items():
        ref = Reference(e.core.reference.coordinates@s.T,np.array([s@q for q in e.core.reference.nodal_triads]))
        law = e.core.section
        section = DirectedHardeningSection(law._elastic,law._direction,law._yield,law._hardening)
        element = NativeP5BeamElement(i,e.node_ids,ref,section,line_force=s@e.core.line_force)
        rotated.add_element(i,element); rotated.materials[element.material_name] = element.core.section
    for boundary in original.boundary_conditions: rotated.add_boundary_condition(deepcopy(boundary))
    program = driver.ArcProgram((.02,.02),2.)
    a = driver.solve_arc_program(original,program); b = driver.solve_arc_program(rotated,program)
    done(a,2); done(b,2)
    expected = a.displacements.reshape(3,6)
    expected = np.c_[expected[:,:3]@s.T,expected[:,3:]@s.T].ravel()
    assert np.linalg.norm(b.displacements-expected) <= 1e-11
    assert abs(a.parameter-b.parameter) <= 1e-11


def test_arc_observer_cannot_mutate_the_metric_program():
    program = driver.ArcProgram((.02,),2.)
    def mutate(event): object.__setattr__(program,'length_scale',3.)
    result = driver.solve_arc_program(problem(),program,progress=mutate)
    assert result.status == 'failed' and 'program changed' in result.failure
    assert result.completed_targets == 0
