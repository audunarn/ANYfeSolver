"""Bounded private integration tests; these do not qualify distributed loads."""
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_fibre_line_work import ReferenceLineForces, evaluate
from anyfesolver import _ge_beam3_fibre_line_program as native
from anyfesolver._ge_beam3_retained_fibre_state import Context as OriginalContext
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from docs.reference_cases import ge_beam3_centered_line_load_oracle as oracle
from test_ge_beam3_spatial_nodal_moments import model as straight_model


def model(*, curved=False, plastic=False, common=None, shift=0.):
    old = straight_model(plastic=plastic, coupled=True)
    section = old.mesh.elements[1].section
    common = np.eye(3) if common is None else common
    nodes = np.array([[0., 0., 0.], [.5, .0625 if curved else 0., 0.], [1., 0., 0.]])
    frames = []
    for x in (0., .5, 1.):
        tangent = np.array([1., .25*(1-2*x) if curved else 0., 0.]); tangent /= np.linalg.norm(tangent)
        second = np.array([0., 0., 1.])
        frames.append(common@np.column_stack((tangent, second, np.cross(tangent, second))))
    made = FEModel('native-reference-line-development')
    for i, row in enumerate(nodes@common.T+shift, 1): made.add_node(i, *row)
    ref = Reference(np.array([n.coords() for n in made.mesh.nodes.values()]), np.array(frames))
    element = NativeRetainedFibreElement(1, (1, 2, 3), ref, section, order=4)
    made.add_element(1, element); made.materials[element.material_name] = section
    made.add_boundary_condition(BoundaryCondition('root', [1], {k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    return made


def line(force=(.03, -.02, .01)):
    return ReferenceLineForces(((1, *map(float, force)),))


@pytest.mark.parametrize('rows', [(), [], ((True,1.,0.,0.),), ((0,1.,0.,0.),),
    ((1,1,0.,0.),), ((1,float('nan'),0.,0.),), ((1,float('inf'),0.,0.),),
    ((1,0.,0.,0.),), ((1,1.,0.,0.),(1,2.,0.,0.)), ((2,1.,0.,0.),(1,2.,0.,0.))])
def test_malformed_pattern_rejected(rows):
    with pytest.raises(ValueError): ReferenceLineForces(rows)


@pytest.mark.parametrize('curved', [False, True])
@pytest.mark.parametrize('scale', [0., 1e-20, 1.])
def test_load_work_matches_separate_rational_geometry_and_closed_derivatives(curved, scale):
    ref = model(curved=curved).mesh.elements[1].operator.reference
    displacement = np.array([[.01,.02,-.03],[.03,.02,.01],[-.02,.01,.04]])
    cells = np.array([rotation([.4,-.2,.3]), rotation([-.1,.35,.2])])
    f = np.array([.03,-.02,.01])*scale
    # Exactly represent the supplied nodal displacement, including low parts.
    from anyfesolver._ge_beam3_p5_coordinates.positions import from_total
    total = np.zeros((3,6)); total[:,:3] = displacement
    high, low = from_total(ref.coordinates, total.ravel(), ref.coordinates+displacement)
    a = evaluate(ref, high, low, cells, f, order=4)
    b = oracle.evaluate(ref.coordinates, displacement, cells, f, order=4)
    assert abs(a.value-b.work) <= 1e-11*max(abs(b.work), np.finfo(float).tiny)
    for actual, expected in ((a.gradient[:36], b.force), (a.hessian[:36,:36], b.hessian)):
        assert np.linalg.norm(actual-expected) <= 1e-11*max(np.linalg.norm(expected), np.finfo(float).tiny)
    assert not np.any(a.gradient[24:]) and not np.any(a.hessian[24:])
    assert not np.any(a.gradient[:18].reshape(3,6)[:,3:])
    if curved and scale:
        assert np.linalg.norm(a.gradient[18:24]) > 0.


def test_complete_loaded_spatial_tangent_includes_internal_cell_load_work():
    p = ForceProgram((.7,), ((3,.01,-.02,.03),))
    c = native.Context(model(curved=True), p, line_forces=line())
    d = np.zeros(c.layout.count); d[c.layout.free] = .02*np.sin(np.arange(len(c.layout.free)))
    state = c.layout.advance(c.initial.mechanical, d)
    r,h,_,_ = c.assemble(state,.7,c.initial.histories)
    old = OriginalContext(model(curved=True),p)
    r0,h0,_,_ = old.assemble(state,.7,old.initial.histories)
    w = c.line_work(state)
    np.testing.assert_allclose(r0-r,.7*w.gradient,atol=1e-14,rtol=1e-11)
    np.testing.assert_allclose(h0-h,.7*w.hessian,atol=1e-14,rtol=1e-11)
    assert np.linalg.norm(w.gradient[18:24]) > 1e-5
    assert np.linalg.norm(h-h.T) < 1e-11
    tangent = c.tangent(r,h,.7)
    for offset in (0., .6, 1.2, 1.8):
        direction = np.zeros(c.layout.count)
        direction[c.layout.free] = np.cos(np.arange(len(c.layout.free))+offset)
        direction /= np.linalg.norm(direction)
        plus = c.assemble(c.layout.advance(state,1e-6*direction),.7,c.initial.histories)[0]
        minus = c.assemble(c.layout.advance(state,-1e-6*direction),.7,c.initial.histories)[0]
        observed = (plus-minus)/2e-6
        assert np.linalg.norm(observed-tangent@direction) <= 1e-7*max(1.,np.linalg.norm(observed))
    with pytest.raises(ValueError, match='external Hessian authority'): c.require_conservative_spectrum()


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    p = ForceProgram((.25,.5,1.,.5,0.,-.5,0.), (), max_iterations=24)
    f = line((.7,.02,.03))
    result = native.solve_force_program(model(plastic=True),p,line_forces=f)
    root = tmp_path_factory.mktemp('line-plastic')
    (root/'whole.json').write_bytes(result.checkpoint)
    (root/'status.json').write_bytes(canonical(dict(status=result.status, failure=result.failure)))
    assert result.status == 'completed', result.failure
    return p,f,result


def test_physical_plasticity_restart_unload_and_replay(accepted,tmp_path):
    p,f,whole = accepted
    assert any(row[2] > 0. for cell in whole.state.histories for station in cell.stations for row in station.rows)
    paused = native.solve_force_program(model(plastic=True),p,line_forces=f,stop_after=3)
    resumed = native.solve_force_program(model(plastic=True),p,line_forces=f,checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    (tmp_path/'paused.json').write_bytes(paused.checkpoint)
    (tmp_path/'resumed.json').write_bytes(resumed.checkpoint)
    assert paused.status == 'paused' and resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == whole.checkpoint
    context = native.Context(model(plastic=True),p,line_forces=f)
    state,records = context.restore(whole.checkpoint)
    assert context.checkpoint(records) == whole.checkpoint
    assert context.recover(state)
    for record in json.loads(whole.checkpoint)['records']:
        reaction = np.array(record['residual'][:18]).reshape(3,6)[0,:3]
        np.testing.assert_allclose(reaction,-record['parameter']*np.array(f.rows[0][1:]),rtol=1e-11,atol=1e-11)
    assert not whole.production_qualified


def test_cancellation_before_commit_preserves_accepted_history(accepted):
    from anyfesolver.control import CancellationToken
    p,f,_ = accepted
    paused = native.solve_force_program(model(plastic=True),p,line_forces=f,stop_after=1)
    token = CancellationToken()
    def cancel(event):
        if event['stage'] == 'fibre-line.before_commit': token.cancel()
    result = native.solve_force_program(model(plastic=True),p,line_forces=f,
        checkpoint=paused.checkpoint,cancellation_token=token,progress=cancel)
    assert result.status == 'cancelled' and result.checkpoint == paused.checkpoint


def test_curved_load_reactions_covariance_and_large_common_rotation(tmp_path):
    p = ForceProgram((.5,1.), (), max_iterations=24); f = np.array([.03,-.02,.01])
    a = native.solve_force_program(model(curved=True),p,line_forces=line(f))
    common = rotation([2.5,-.7,.8])
    b = native.solve_force_program(model(curved=True,common=common),p,line_forces=line(common@f))
    (tmp_path/'curved.json').write_bytes(a.checkpoint)
    (tmp_path/'curved-rotated.json').write_bytes(b.checkpoint)
    assert a.status == b.status == 'completed', (a.failure,b.failure)
    np.testing.assert_allclose(b.state.mechanical.positions,a.state.mechanical.positions@common.T,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(b.state.mechanical.nodal_frames,common@a.state.mechanical.nodal_frames,rtol=1e-11,atol=1e-11)
    # Rigid virtual work checks both force and moment equilibrium on the actual
    # deformed lifted line, including the internal cell-rotation load terms.
    c = native.Context(model(curved=True),p,line_forces=line(f)); state,_ = c.restore(a.checkpoint)
    w = c.line_work(state.mechanical); r = json.loads(a.checkpoint)['records'][-1]['residual']
    np.testing.assert_allclose(np.array(r[:3]),-w.gradient[:18].reshape(3,6)[:,:3].sum(axis=0),atol=1e-11)
    x = state.mechanical.positions
    total_moment = np.cross(x,w.gradient[:18].reshape(3,6)[:,:3]).sum(axis=0)+w.gradient[18:24].reshape(2,3).sum(axis=0)
    np.testing.assert_allclose(np.array(r[3:6])+np.cross(x[0],r[:3]),-total_moment,atol=1e-11)


def test_authority_mutation_and_cross_program_restart_rejected(accepted):
    p,f,result = accepted
    with pytest.raises(ValueError): OriginalContext(model(plastic=True),p).restore(result.checkpoint)
    with pytest.raises(ValueError): native.Context(model(plastic=True),p,line_forces=line((.71,.02,.03))).restore(result.checkpoint)
    packet = json.loads(result.checkpoint); packet['program']['line_forces']['measure']='CURRENT_ARCLENGTH'
    packet['checkpoint_sha256'] = sha({k:v for k,v in packet.items() if k != 'checkpoint_sha256'})
    context = native.Context(model(plastic=True),p,line_forces=f)
    with pytest.raises(ValueError): context.restore(canonical(packet))
    context.program_data['line_search']='changed'
    with pytest.raises(ValueError,match='frozen'): context.guard()
    pattern=line(); context=native.Context(model(),ForceProgram((0.,)),line_forces=pattern)
    object.__setattr__(pattern,'rows',((1,.1,0.,0.),))
    with pytest.raises(ValueError,match='frozen'): context.guard()
    with pytest.raises(ValueError,match='absent'): native.Context(model(),p,line_forces=ReferenceLineForces(((2,1.,0.,0.),)))


def test_failed_step_has_no_trial_history_publication():
    p=ForceProgram((1.,),(),max_iterations=0)
    c=native.Context(model(),p,line_forces=line())
    result=native.solve_force_program(model(),p,line_forces=line())
    assert result.status=='failed' and result.completed_targets==0
    assert result.checkpoint==c.checkpoint(())


def test_exact_large_translation_and_connectivity_reversal_preserve_line_work():
    from anyfesolver._ge_beam3_p5_coordinates.positions import from_total
    displacement=np.array([[.03125,.0625,-.125],[.0625,.03125,.0625],[-.125,.0625,.03125]])
    total=np.zeros((3,6)); total[:,:3]=displacement
    cells=np.array([rotation([.4,-.2,.3]),rotation([-.1,.35,.2])])
    force=np.array([.03,-.02,.01]); packets=[]
    for shift in (0.,2.**30,2.**40):
        ref=model(curved=True,shift=shift).mesh.elements[1].operator.reference
        high,low=from_total(ref.coordinates,total.ravel(),ref.coordinates+displacement)
        work=evaluate(ref,high,low,cells,force,order=4)
        packets.append(canonical(work))
    assert packets[0]==packets[1]==packets[2]
    ref=model(curved=True).mesh.elements[1].operator.reference
    reversed_ref=Reference(ref.coordinates[::-1],ref.nodal_triads[::-1]@np.diag([-1.,1.,-1.]))
    high,low=from_total(ref.coordinates,total.ravel(),ref.coordinates+displacement)
    a=evaluate(ref,high,low,cells,force,order=4)
    b=evaluate(reversed_ref,high[::-1],low[::-1],cells[::-1],force,order=4)
    indices=np.r_[np.arange(12,18),np.arange(6,12),np.arange(6),np.arange(21,24),np.arange(18,21),np.arange(24,42)]
    np.testing.assert_allclose(b.value,a.value,rtol=1e-11,atol=1e-14)
    np.testing.assert_allclose(b.gradient,a.gradient[indices],rtol=1e-11,atol=1e-14)
    np.testing.assert_allclose(b.hessian,a.hessian[np.ix_(indices,indices)],rtol=1e-11,atol=1e-14)


def test_disable_load_path_and_unsupported_modes_are_rejected():
    c=native.Context(model(),ForceProgram((0.,)),line_forces=line())
    c._line_ready=False
    with pytest.raises(ValueError,match='assembly disabled'): c.guard()
    ref=model().mesh.elements[1].operator.reference
    for order in (True,5,24):
        with pytest.raises(ValueError): evaluate(ref,ref.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)),[1.,0.,0.],order=order)
