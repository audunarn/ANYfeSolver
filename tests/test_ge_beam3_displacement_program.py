"""Small native control checks, not post-buckling qualification."""

import json

import numpy as np
import pytest
from scipy import sparse

import anyfesolver._ge_beam3_displacement_program as driver
from anyfesolver._ge_beam3_load_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_load_state import problem, store_for, assemble


def completed(result,count):
    assert result.status == 'completed', result.failure
    assert result.completed_targets == count and not result.production_qualified
    assert np.linalg.norm(result.physical_imbalance[6:]) <= 1e-11


def reseal(value):
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    return canonical({**body,'checkpoint_sha256':sha(body)})


def test_full_border_can_cross_a_simple_singular_stiffness():
    matrix = sparse.diags([0.,2.],format='csr'); column = np.array([-1.,0.])
    border = driver._bordered(matrix,column,np.array([0,1]),0).toarray()
    assert np.linalg.matrix_rank(matrix.toarray()) == 1
    assert np.linalg.matrix_rank(border) == 3
    increment = np.linalg.solve(border,np.array([0.,0.,.1]))
    np.testing.assert_array_equal(increment,[.1,0.,0.])
    # This algebraic example is not evidence of any beam's equilibrium branch.


def test_elastic_displacement_target_recovers_force_parameter():
    fixed = solve_force_program(problem(),ForceProgram((.5,)))
    completed(fixed,1)
    target = float(fixed.displacements[12])
    result = driver.solve_displacement_program(problem(),driver.DisplacementProgram((target,),55,'ux'))
    completed(result,1)
    assert abs(result.parameter-.5) <= 1e-9
    np.testing.assert_allclose(result.displacements,fixed.displacements,atol=1e-10,rtol=0.)
    assert abs(result.displacements[12]-target) <= 1e-11


def test_plastic_displacement_reversal_split_restart_exact():
    program = driver.DisplacementProgram((.02,.04,0.,-.02),55,'ux')
    full = driver.solve_displacement_program(problem(plastic=True),program)
    completed(full,4)
    paused = driver.solve_displacement_program(problem(plastic=True),program,stop_after=2)
    assert paused.status == 'paused'
    resumed = driver.solve_displacement_program(problem(plastic=True),program,checkpoint=paused.checkpoint)
    completed(resumed,4)
    assert full.checkpoint == resumed.checkpoint
    value = json.loads(full.checkpoint)
    assert value['records'][-1]['displacement_target'] == -.02
    inner = json.loads(value['element_states'][0]['state']['payload'])['material_state']
    assert any(h['accumulated'] > 0. for h in inner['histories'])


@pytest.mark.parametrize('plastic',[False,True])
def test_assembled_parameter_column_matches_finite_difference(plastic):
    from test_ge_beam3_curved_p5_native_chart_probe import sample
    model = problem(plastic=plastic,count=1); store = store_for(model)
    nodal = np.zeros(18); nodal[12:15] = [.025,-.01,.005]
    total = sample(); step = 1e-5
    try:
        _,_,trial = assemble(model,store,total,.7)
        column = driver._parameter_column(model,tuple(sorted(model.mesh.elements.items())),store,trial,18,nodal)
        plus,_,_ = assemble(model,store,total,.7+step)
        minus,_,_ = assemble(model,store,total,.7-step)
        expected = (plus-minus)/(2*step)-nodal
        assert np.linalg.norm(column-expected) <= 1e-7
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())


@pytest.mark.parametrize('stage',['native_displacement.before_commit','native_displacement.committed'])
def test_cancellation_retains_last_accepted_control_step(stage):
    program = driver.DisplacementProgram((.005,.01),55,'ux'); token = CancellationToken()
    count = 0 if stage.endswith('before_commit') else 1
    def cancel(event):
        if event['stage'] == stage: token.cancel('control fixture')
    result = driver.solve_displacement_program(problem(),program,cancellation_token=token,progress=cancel)
    assert result.status == 'cancelled' and result.completed_targets == count
    retained = driver.solve_displacement_program(problem(),program,checkpoint=result.checkpoint,stop_after=count)
    assert retained.checkpoint == result.checkpoint


def test_failed_control_step_does_not_advance_history():
    program = driver.DisplacementProgram((.005,),55,'ux',max_iterations=0)
    virgin = driver.solve_displacement_program(problem(),program,stop_after=0)
    result = driver.solve_displacement_program(problem(),program)
    assert result.status == 'failed' and 'iteration bound' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint


@pytest.fixture(scope='module')
def accepted():
    program = driver.DisplacementProgram((.005,),55,'ux')
    result = driver.solve_displacement_program(problem(),program)
    completed(result,1)
    return program,result.checkpoint


@pytest.mark.parametrize('kind',['control','parameter','reaction','record','hash','force_schema'])
def test_resealed_control_restart_mutation_rejected(accepted,kind):
    program,raw = accepted; value = json.loads(raw)
    if kind == 'control': value['total_displacement'][12] += .01
    elif kind == 'parameter': value['accepted_parameter'] += .1
    elif kind == 'reaction': value['physical_imbalance'][0] += .1
    elif kind == 'record': value['records'][0]['displacement_target'] = .01
    elif kind == 'hash': value['model_sha256'] = '0'*64
    elif kind == 'force_schema': value['schema'] = 'GE_BEAM3_NATIVE_FORCE_PROGRAM_CHECKPOINT_V1'
    with pytest.raises(ValueError):
        driver.solve_displacement_program(problem(),program,checkpoint=reseal(value))


@pytest.mark.parametrize('node,component',[(True,'ux'),(55,'rx'),(55,None),(55,1)])
def test_rotation_or_implicit_control_rejected(node,component):
    with pytest.raises(ValueError): driver.DisplacementProgram((.01,),node,component)


@pytest.mark.parametrize('node',[7,999])
def test_constrained_or_absent_control_rejected_before_assembly(monkeypatch,node):
    def forbidden(*args,**kwargs): raise AssertionError('must not assemble')
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',forbidden)
    with pytest.raises(ValueError):
        driver.solve_displacement_program(problem(),driver.DisplacementProgram((.01,),node,'ux'))


def arch():
    """Two-element h=.1 specimen; same section as the preserved arch probe.

    Reconstruct directly with native elements, not the old research mechanics.
    No claim that this coarse branch is an engineering/stability certificate.
    """
    from anyfesolver.fe_core import FEModel
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
    made = FEModel('native-displacement-arch'); ids = (7,23,55,81,103)
    x = np.linspace(-1.,1.,5); points = np.array([[s,.1*(1-s*s),0.] for s in x])
    frames = []
    for s in x:
        first = np.array([1.,-.2*s,0.]); first /= np.linalg.norm(first)
        second = np.array([0.,0.,1.]); frames.append(np.column_stack((first,second,np.cross(first,second))))
    for i,point in zip(ids,points): made.add_node(i,*point)
    for cell in (0,1):
        ref = Reference(points[2*cell:2*cell+3],np.array(frames[2*cell:2*cell+3]))
        section = DirectedHardeningSection(np.diag([1000.,400.,400.,.02,.01,.02]),np.array([1.,0.,0.,0.,0.,0.]),1e6,1.)
        element = NativeP5BeamElement(cell+1,ids[2*cell:2*cell+3],ref,section,line_force=np.zeros(3))
        made.add_element(cell+1,element); made.materials[element.material_name] = element.core.section
    made.add_boundary_condition(BoundaryCondition('ends',[7,103],{s:0. for s in ('ux','uy','uz','rx','ry','rz')}))
    return made


def test_native_arch_displacement_control_reaches_decreasing_load_branch(tmp_path):
    program = driver.DisplacementProgram((-.02,-.04,-.06,-.08),55,'uy',((55,0.,-1.,0.),),max_iterations=16)
    events = []
    result = driver.solve_displacement_program(arch(),program,progress=events.append)
    # External test diagnostics only, including a last-accepted capsule if the
    # new driver fails; never a fabricated canonical qualification record.
    (tmp_path/'diagnostic-checkpoint.json').write_bytes(result.checkpoint)
    (tmp_path/'diagnostic-status.json').write_bytes(canonical(dict(status=result.status,
        failure=result.failure,completed_targets=result.completed_targets,production_qualified=False)))
    (tmp_path/'diagnostic-progress.json').write_bytes(canonical(events))
    assert result.status == 'completed', (result.failure,[e for e in events if e['stage'].endswith('.iteration')])
    assert result.completed_targets == 4
    value = json.loads(result.checkpoint); loads = [r['parameter'] for r in value['records']]
    assert all(e['control_error'] == 0. for e in events if e['stage'].endswith('.iteration'))
    assert any(right < left for left,right in zip(loads,loads[1:])), loads
    assert abs(result.displacements[13]+.08) <= 1e-11
    assert np.linalg.norm(result.physical_imbalance[6:24]) <= 1e-11
    assert not result.production_qualified


def test_serialization_failure_preserves_accepted_displacement_capsule(monkeypatch):
    program = driver.DisplacementProgram((.005,),55,'ux')
    virgin = driver.solve_displacement_program(problem(),program,stop_after=0)
    original = driver._capsule
    def reject(*args):
        if args[-3]: raise ValueError('control capsule staging fixture')
        return original(*args)
    monkeypatch.setattr(driver,'_capsule',reject)
    result = driver.solve_displacement_program(problem(),program)
    assert result.status == 'failed' and 'capsule staging fixture' in result.failure
    assert result.checkpoint == virgin.checkpoint and result.completed_targets == 0


def test_zero_load_column_fails_without_regularization_or_retry():
    program = driver.DisplacementProgram((.005,),55,'ux'); attempts = []
    def progress(event):
        if event['stage'] == 'native_displacement.before_assembly': attempts.append(event['target'])
    result = driver.solve_displacement_program(problem(line_force=(0.,0.,0.)),program,progress=progress)
    assert result.status == 'failed' and result.completed_targets == 0
    assert attempts == [1]


@pytest.mark.parametrize('kind',['duplicate','nonfinite','whitespace','missing'])
def test_noncanonical_displacement_capsules_rejected(accepted,kind):
    program,raw = accepted
    if kind == 'duplicate': raw = raw[:-2]+b',"schema":"duplicate"}\n'
    elif kind == 'nonfinite': raw = b'{"bad":Infinity}\n'
    elif kind == 'whitespace': raw += b' '
    elif kind == 'missing': raw = b'{}\n'
    with pytest.raises(ValueError): driver.solve_displacement_program(problem(),program,checkpoint=raw)


def test_control_mutation_is_detected_before_assembly():
    program = driver.DisplacementProgram((.005,),55,'ux')
    def mutate(event): object.__setattr__(program,'control_component','uy')
    result = driver.solve_displacement_program(problem(),program,progress=mutate)
    assert result.status == 'failed' and 'program changed' in result.failure
    assert result.completed_targets == 0


def test_control_deadline_preserves_virgin_checkpoint(monkeypatch):
    from types import SimpleNamespace
    program = driver.DisplacementProgram((.005,),55,'ux')
    virgin = driver.solve_displacement_program(problem(),program,stop_after=0)
    clocks = iter((0.,601.))
    monkeypatch.setattr(driver,'time',SimpleNamespace(monotonic=lambda: next(clocks)))
    result = driver.solve_displacement_program(problem(),program)
    assert result.status == 'failed' and 'deadline' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint
