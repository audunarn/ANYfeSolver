"""Small native moment-load development tests; no public qualification."""
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection, Fibre, FlowCurve
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver import _ge_beam3_spatial_fibre_program as native
from anyfesolver._ge_beam3_retained_fibre_state import Layout, Context as OriginalContext
from anyfesolver._ge_beam3_seeded_fibre_control import spatial_jacobian
from anyfesolver._ge_beam3_p5.algebra import rotation, log_rotation, skew
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha


def model(common=None, *, plastic=False, coupled=False):
    common = np.eye(3) if common is None else common
    made = FEModel('private-moment-beam')
    for i, x in enumerate((0., .5, 1.), 1): made.add_node(i, *(common@np.array([x, 0., 0.])))
    curve = FlowCurve.linear(.25 if plastic else 1e6, 2.)
    fibres = tuple(Fibre(str(i), y, z, .25, 16., curve)
        for i, (y, z) in enumerate(((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))))
    factor = np.zeros((3, 6)); factor[0, 1] = np.sqrt(10.); factor[1, 2] = np.sqrt(12.); factor[2, 3] = np.sqrt(2.)
    if coupled:
        factor[0, 0] = .2; factor[1, 4] = .3; factor[2, 5] = -.2
    section = PhysicalFibreSection(fibres, factor)
    ref = Reference(np.array([n.coords() for n in made.mesh.nodes.values()]), np.tile(common, (3,1,1)))
    element = NativeRetainedFibreElement(1, (1,2,3), ref, section, order=4)
    made.add_element(1, element); made.materials[element.material_name] = section
    made.add_boundary_condition(BoundaryCondition('root', [1], {k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    return made


@pytest.mark.parametrize('rows', [(), [(3,1.,0.,0.)], ((True,1.,0.,0.),), ((0,1.,0.,0.),),
    ((3,1,0.,0.),), ((3,float('nan'),0.,0.),), ((3,True,0.,0.),), ((3,0.,0.,0.),),
    ((3,1.,0.,0.),(3,2.,0.,0.)), ((3,1.,0.,0.),(2,2.,0.,0.))])
def test_malformed_moment_pattern_rejected(rows):
    with pytest.raises(ValueError): SpatialNodalMoments(rows)


def test_constant_spatial_couple_is_not_a_fictitious_log_potential():
    moments = SpatialNodalMoments(((3,0.,0.,1.),))
    with pytest.raises(ValueError, match='no general conservative'): moments.potential(np.eye(3))
    steps = ([.2,0.,0.], [0.,.3,0.], [-.2,0.,0.], [0.,-.3,0.])
    q = np.eye(3)
    for step in steps: q = rotation(step)@q
    closing = -log_rotation(q)
    np.testing.assert_allclose(rotation(closing)@q, np.eye(3), atol=1e-14)
    work = float(np.array([0.,0.,1.])@(np.sum(steps,axis=0)+closing))
    assert abs(work) > .01  # Closed orientation path with nonzero applied work.
    assert not moments.descriptor()['conservative_potential']


def test_default_capture_preserved_and_moments_bound_to_nodes():
    made = model(); p = ForceProgram((0.,), ((3,.1,.2,.3),))
    plain = Layout(made,p); explicit = Layout(made,p,nodal_moments=None)
    assert plain.identity == explicit.identity
    np.testing.assert_array_equal(plain.force.reshape(-1,6)[:,3:], 0.)
    moments = SpatialNodalMoments(((3,.01,.02,.03),)); loaded = Layout(made,p,nodal_moments=moments)
    assert plain.identity != loaded.identity
    np.testing.assert_array_equal(loaded.force[-6:], [.1,.2,.3,.01,.02,.03])
    with pytest.raises(ValueError, match='absent'): Layout(made,p,nodal_moments=SpatialNodalMoments(((4,1.,0.,0.),)))
    object.__setattr__(moments, 'rows', ((3,.02,.02,.03),))
    with pytest.raises(ValueError, match='frozen'): loaded.guard()


def test_spatial_tangent_uses_internal_not_net_moments():
    p = ForceProgram((.7,), ((3,.02,-.03,.04),))
    moments = SpatialNodalMoments(((3,.03,.04,-.05),))
    c = native.Context(model(coupled=True),p,nodal_moments=moments)
    initial = np.zeros(c.layout.count)
    initial[c.layout.free] = .01*np.sin(np.arange(len(c.layout.free)))
    state = c.layout.advance(c.initial.mechanical,initial)
    r,h,_,_ = c.assemble(state,.7,c.initial.histories)
    correct = c.tangent(r,h,.7); wrong = spatial_jacobian(c.layout,r,h)
    expected = np.zeros_like(h); expected[15:18,15:18] = -.35*skew([.03,.04,-.05])
    np.testing.assert_allclose(correct-wrong,expected,atol=1e-14,rtol=1e-13)
    errors = []
    for offset in (0.,.6,1.2,1.8):
        direction = np.zeros(c.layout.count); direction[c.layout.free] = np.cos(np.arange(len(c.layout.free))+offset)
        direction /= np.linalg.norm(direction); step = 1e-6
        plus = c.assemble(c.layout.advance(state,step*direction),.7,c.initial.histories)[0]
        minus = c.assemble(c.layout.advance(state,-step*direction),.7,c.initial.histories)[0]
        observed = (plus-minus)/(2*step)
        errors.append(np.linalg.norm(observed-correct@direction)/max(1.,np.linalg.norm(observed)))
    assert max(errors) < 1e-7
    with pytest.raises(ValueError, match='spectral authority'): c.require_conservative_spectrum()


def test_pure_torsion_large_rotation_unload_reverse_and_restart(tmp_path):
    p = ForceProgram((.3,.6,1.2,.6,0.,-.6,0.), (), max_iterations=24)
    moments = SpatialNodalMoments(((3,2.,0.,0.),))  # GJ=2, L=1: tip angle=parameter.
    whole = native.solve_force_program(model(),p,nodal_moments=moments)
    (tmp_path/'torsion.json').write_bytes(whole.checkpoint)
    assert whole.status == 'completed', whole.failure
    rows = json.loads(whole.checkpoint)['records']
    for row in rows:
        angle = row['parameter']; q = np.array(row['mechanical']['nodal_frames'])
        np.testing.assert_allclose(q[-1],rotation([angle,0.,0.]),atol=1e-11,rtol=1e-11)
        np.testing.assert_allclose(row['mechanical']['positions'], [[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]],atol=1e-11)
        reaction = np.array(row['residual'][:18]).reshape(3,6)
        assert np.linalg.norm(reaction[0,3:]+[2*angle,0.,0.]) < 1e-11
    paused = native.solve_force_program(model(),p,nodal_moments=moments,stop_after=3)
    resumed = native.solve_force_program(model(),p,nodal_moments=moments,checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    assert paused.status == 'paused' and resumed.status == 'completed'
    assert resumed.checkpoint == whole.checkpoint
    c = native.Context(model(),p,nodal_moments=moments); accepted,_ = c.restore(whole.checkpoint)
    assert c.recover(accepted)
    with pytest.raises(ValueError): native.Context(model(),p).restore(whole.checkpoint)
    with pytest.raises(ValueError): OriginalContext(model(),p).restore(whole.checkpoint)
    assert not whole.production_qualified


def test_biaxial_bending_moment_and_common_large_rotation_covariance(tmp_path):
    p = ForceProgram((.5,1.), (), max_iterations=24); moment=np.array([0.,.6,.8])
    moments = SpatialNodalMoments(((3,*map(float,moment)),))
    a = native.solve_force_program(model(),p,nodal_moments=moments)
    (tmp_path/'bending.json').write_bytes(a.checkpoint)
    assert a.status == 'completed',a.failure
    # Equal EI=4: constant-axis bending rotation is exactly M L/EI.
    np.testing.assert_allclose(a.state.mechanical.nodal_frames[-1],rotation(moment/4),rtol=1e-11,atol=1e-11)
    common=rotation([2.5,-.7,.8]); rotated=SpatialNodalMoments(((3,*map(float,common@moment)),))
    b = native.solve_force_program(model(common),p,nodal_moments=rotated)
    (tmp_path/'bending-rotated.json').write_bytes(b.checkpoint)
    assert b.status == 'completed',b.failure
    np.testing.assert_allclose(b.state.mechanical.positions,(common@a.state.mechanical.positions.T).T,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(b.state.mechanical.nodal_frames,common@a.state.mechanical.nodal_frames,atol=1e-11,rtol=1e-11)


@pytest.fixture(scope='module')
def coupled_plastic(tmp_path_factory):
    p = ForceProgram((.25,.5,1.,.5,0.,-.5,0.), ((3,.5,.02,.03),), max_iterations=24)
    moments = SpatialNodalMoments(((3,.2,.15,.1),))
    result = native.solve_force_program(model(plastic=True,coupled=True),p,nodal_moments=moments)
    root = tmp_path_factory.mktemp('coupled-plastic-moments')
    (root/'whole.json').write_bytes(result.checkpoint)
    (root/'status.json').write_bytes(canonical(dict(status=result.status,failure=result.failure)))
    assert result.status == 'completed',result.failure
    return p,moments,result


def test_all_six_load_components_physical_plasticity_reactions_and_restart(coupled_plastic,tmp_path):
    p,moments,whole=coupled_plastic
    assert any(v[2] > 0. for cell in whole.state.histories for station in cell.stations for v in station.rows)
    for row in json.loads(whole.checkpoint)['records']:
        parameter=row['parameter']; residual=np.array(row['residual'][:18]).reshape(3,6)
        applied=np.zeros((3,6)); applied[-1,:3]=parameter*np.array(p.nodal_forces[0][1:])
        applied[-1,3:]=parameter*np.array(moments.rows[0][1:])
        positions=np.array(row['mechanical']['positions'])+np.array(row['mechanical']['position_low'])
        assert np.linalg.norm(np.sum(residual[:,:3]+applied[:,:3],axis=0)) < 1e-11
        moment=np.sum(residual[:,3:]+applied[:,3:]+np.cross(positions,residual[:,:3]+applied[:,:3]),axis=0)
        assert np.linalg.norm(moment) < 1e-11
    paused=native.solve_force_program(model(plastic=True,coupled=True),p,nodal_moments=moments,stop_after=3)
    resumed=native.solve_force_program(model(plastic=True,coupled=True),p,nodal_moments=moments,
        checkpoint=paused.checkpoint,expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    (tmp_path/'paused.json').write_bytes(paused.checkpoint)
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint


def test_cancellation_before_commit_preserves_last_accepted_capsule(coupled_plastic):
    from anyfesolver.control import CancellationToken
    p,moments,_=coupled_plastic; token=CancellationToken()
    paused=native.solve_force_program(model(plastic=True,coupled=True),p,nodal_moments=moments,stop_after=1)
    def cancel(event):
        if event['stage']=='spatial-fibre.before_commit': token.cancel()
    result=native.solve_force_program(model(plastic=True,coupled=True),p,nodal_moments=moments,
        checkpoint=paused.checkpoint,cancellation_token=token,progress=cancel)
    assert result.status == 'cancelled' and result.checkpoint == paused.checkpoint


def test_moment_authority_and_program_mutations_do_not_replay(coupled_plastic):
    p,moments,result=coupled_plastic
    other=SpatialNodalMoments(((3,.21,.15,.1),))
    with pytest.raises(ValueError): native.Context(model(plastic=True,coupled=True),p,nodal_moments=other).restore(result.checkpoint)
    data=json.loads(result.checkpoint)
    data['program']['nodal_moments']['axes']='FOLLOWER_MATERIAL'
    data['checkpoint_sha256']=sha({k:v for k,v in data.items() if k!='checkpoint_sha256'})
    context=native.Context(model(plastic=True,coupled=True),p,nodal_moments=moments)
    with pytest.raises(ValueError): context.restore(canonical(data))
    context.program_data['line_search']='wrong'
    with pytest.raises(ValueError,match='capture changed'): context.guard()


def test_existing_six_macro_checkpoint_replays_without_new_equilibrium_solve():
    from pathlib import Path
    from docs.reference_cases import ge_beam3_fibre_arch_refinement as family
    from anyfesolver import _ge_beam3_seeded_fibre_control as old
    path=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-fibre-arch6-20260907-v1/checkpoint-diagnostic.json')
    if not path.is_file(): pytest.skip('Preserved external development checkpoint not installed; not qualification')
    raw=path.read_bytes()
    assert sha256(raw).hexdigest()=='20a1a48cf6e2dfdc669c4f2d823deb89a9da8c6560ab716eb32e95b1b917a940'
    context=old.Context(family.model(6),family.program(6))
    state,records=context.restore(raw)
    assert state.completed_targets==4 and context.checkpoint(records)==raw
