"""Private nodal dead-force work, shared-node equilibrium and state parity."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_nodal_loading as nodal
from anyfesolver._ge_beam3_retained_generalized_state import Context,Program
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_schur_line_program import save

EMPTY=DistributedPattern(LinePattern(()),())
FORCE=(.005,-.003,.002)


def program(force=FORCE,*,iterations=24):
    return nodal.Program((.5,1.),EMPTY,nodal.NodalDeadForces(((3,*force),)),max_iterations=iterations)


def resume(m,p,result,**kw):
    return nodal.solve(m,p,checkpoint=result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest(),**kw)


@pytest.mark.parametrize('macros,curved',((1,False),(1,True),(2,True)))
def test_exact_load_work_and_tangent(macros,curved,tmp_path):
    m,_,_=make(curved,macros,clamped=True)
    line=DistributedPattern(LinePattern(tuple((eid,.001,-.002,.003) for eid in m.mesh.elements)),())
    p=nodal.Program((1.,),line,nodal.NodalDeadForces(((3,*FORCE),)))
    c=nodal.Context(m,p);old=Context(m,Program((1.,),line))
    step=np.sin(np.arange(c.count)+.3)*.001;step[:6]=0.
    state=c.advance(c.initial.mechanical,step)
    r,j,_,_,work=c.assemble(state,.7,c.initial.histories)
    r0,j0,_,_,w0=old.assemble(state,.7,old.initial.histories)
    external=c.nodal_external(.7)
    assert np.linalg.norm(r-r0+np.r_[external,np.zeros(c.count-c.nodal_count)])<=1e-11
    np.testing.assert_array_equal(j,j0);assert work[:-1]==w0
    direction=np.cos(np.arange(c.count)+.2)*.0001;direction[:6]=0.
    changed=c.advance(state,direction)
    error=abs(c.nodal_work(changed,.7)-c.nodal_work(state,.7)-external@direction[:c.nodal_count])
    assert error<=1e-11
    assert np.count_nonzero(external)==3 and not np.any(external.reshape(-1,6)[:,3:])
    save(tmp_path/'work.json',dict(macros=macros,curved=curved,external=external,work=work,
        derivative_error=error,tangent_unchanged=True,shared_node_counted_once=True))


@pytest.mark.parametrize('macros,curved',((1,False),(1,True),(2,True)))
def test_actual_point_force_restart_and_modal(macros,curved,tmp_path):
    m,_,masses=make(curved,macros,clamped=True);p=program()
    result=nodal.solve(m,p);save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    paused=nodal.solve(m,p,stop_after=1);assert paused.status=='paused',paused.failure
    restarted=resume(m,p,paused);assert restarted.checkpoint==result.checkpoint
    context=nodal.Context(m,p);state,records=context.restore(result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    before=canonical(state);recovery=context.recover(state)
    row=json.loads(records[-1]);reaction=np.array(row['residual'][:6])
    points=state.mechanical.positions+state.mechanical.position_low
    force_error=float(np.linalg.norm(reaction[:3]+FORCE))
    moment_error=float(np.linalg.norm(reaction[3:]+np.cross(points[2]-points[0],FORCE)))
    assert max(force_error,moment_error)<=1e-11
    packet,guard=nodal.prepare(m,p,result.checkpoint,masses,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert np.linalg.norm(packet.net_residual[list(packet.free_dofs)])<=1e-11
    if macros==1 and curved:
        other,modes=nodal.solve_modes(m,p,result.checkpoint,masses,expected_sha256=sha256(result.checkpoint).hexdigest(),bounds=(0.,1e5),num_modes=6)
        assert canonical(other)==canonical(packet) and np.all(modes.eigenvalues>0.)
        save(tmp_path/'modes.json',modes)
    assert canonical(context.recover(state))==canonical(recovery) and canonical(state)==before
    assert context.checkpoint(records)==result.checkpoint
    guard();save(tmp_path/'point.json',dict(packet=packet,recovery=recovery,force_error=force_error,
        moment_error=moment_error,restart_identical=True,production_qualified=False))


@pytest.mark.parametrize('force',(-.1,.1))
def test_analytical_axial(force,tmp_path):
    m,_,masses=make(False,1,clamped=True,coupled=False);p=program((force,0.,0.))
    r=nodal.solve(m,p);assert r.status=='completed',r.failure
    s=r.state.mechanical;reference=m.mesh.elements[1].operator.reference.coordinates
    displacement=s.positions-reference+s.position_low
    expected=np.zeros((3,3));expected[:,0]=force*(reference[:,0]-reference[0,0])/1000.
    assert np.linalg.norm(displacement-expected)<=1e-11
    row=json.loads(r.checkpoint)['records'][-1]
    assert abs(row['residual'][0]+force)<=1e-11
    assert abs(row['work'][-1]-force*expected[-1,0])<=1e-11
    packet,guard=nodal.prepare(m,p,r.checkpoint,masses,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert np.linalg.norm(packet.geometric)>0 and np.linalg.norm(packet.net_residual[list(packet.free_dofs)])<=1e-11
    guard();save(tmp_path/'axial.json',dict(force=force,displacement=displacement,expected=expected,packet=packet,checkpoint_sha256=sha256(r.checkpoint).hexdigest()))


@pytest.mark.parametrize('rows',([((3,1.,0.,0.))],((3,True,0.,0.),),((True,1.,0.,0.),),
    ((3,float('nan'),0.,0.),),((3,1.,0.,0.),(3,1.,0.,0.)),((3,1.,0.,0.),(1,0.,0.,1.)),((3,1.,0.,0.,0.,0.,0.),)))
def test_force_schema(rows):
    with pytest.raises(ValueError):nodal.NodalDeadForces(rows)


@pytest.mark.parametrize('kind',('unknown','cross_schema','work','reaction','loads','cancel','failed'))
def test_state_guards(kind,tmp_path):
    m,_,_=make(True,1,clamped=True);p=program()
    if kind=='unknown':
        with pytest.raises(ValueError,match='unknown'):
            nodal.Context(m,nodal.Program((1.,),EMPTY,nodal.NodalDeadForces(((99,*FORCE),))))
    elif kind=='cancel':
        token=CancellationToken()
        def progress(row):
            if row['stage']=='retained-generalized.before_commit':token.cancel()
        result=nodal.solve(m,p,cancellation_token=token,progress=progress)
        assert result.status=='cancelled' and result.completed_targets==0
        assert result.checkpoint==nodal.Context(m,p).checkpoint(())
    elif kind=='failed':
        result=nodal.solve(m,program(iterations=1));assert result.status=='failed' and result.completed_targets==0
    else:
        result=nodal.solve(m,p,stop_after=1);assert result.status=='paused',result.failure
        if kind=='cross_schema':
            old=Context(m,Program(p.targets,p.pattern))
            with pytest.raises(ValueError):old.restore(result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
            with pytest.raises(ValueError):nodal.Context(m,p).restore(old.checkpoint(()),expected_sha256=sha256(old.checkpoint(())).hexdigest())
        else:
            value=json.loads(result.checkpoint);row=value['records'][0]
            if kind=='work':row['work'][-1]+=.01
            elif kind=='reaction':row['residual'][0]+=.01
            else:value['program']['nodal_forces']['rows'][0][1]+=.01
            row['record_sha256']=sha({k:v for k,v in row.items() if k!='record_sha256'})
            value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'})
            raw=canonical(value)
            with pytest.raises(ValueError):nodal.Context(m,p).restore(raw,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'guard.json',dict(kind=kind,rejected_or_rolled_back=True))
