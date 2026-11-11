"""Authenticated retained current-rest capture, not continuum qualification."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_generalized_modal as modal
from anyfesolver import _ge_beam3_native_generalized_factor_modal as prior
from anyfesolver._ge_beam3_retained_generalized_state import Context,Program
from anyfesolver._ge_beam3_retained_generalized_program import solve
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver.control import CancellationToken,SolveCancelled
from test_ge_beam3_generalized_slenderness_diagnostic import make
from test_ge_beam3_native_generalized_modal import make as connected
from test_ge_beam3_schur_line_program import save

EMPTY=DistributedPattern(LinePattern(()),())


def capture(m,p,raw,masses,**kw):
    return modal.prepare(m,p,raw,masses,expected_sha256=sha256(raw).hexdigest(),**kw)


@pytest.mark.parametrize('rho,curved',((100.,False),(100.,True),(10000.,False),(10000.,True),(1000000.,False),(1000000.,True)))
def test_virgin_factor_port(rho,curved,tmp_path):
    for label,q in (('E',np.eye(3)),('GENERAL',rotation([.4,-.3,.2]))):
        print(dict(stage='capture',rho=rho,curved=curved,frame=label),flush=True)
        m,states,masses=make(rho,curved,q);p=Program((0.,),EMPTY)
        context=Context(m,p);raw=context.checkpoint(())
        before=canonical(states)
        old,old_guard=prior.prepare(m,states,np.zeros(18),masses,np.zeros(18))
        packet,guard=capture(m,p,raw,masses)
        for key in ('left','right','geometric','kinetic'):
            np.testing.assert_array_equal(getattr(packet,key),getattr(old,key))
        np.testing.assert_array_equal(packet.mass,old.base.mass)
        assert packet.free_dofs==old.base.free_dofs and packet.algebraic_dofs==old.base.algebraic_dofs
        assert packet.internal_layout==old.base.internal_layout
        assert canonical(states)==before and context.checkpoint(())==raw
        assert not packet.production_qualified and not packet.buckling_factor_authorized
        for key in ('left','right','geometric','kinetic','stiffness','mass','net_residual'):
            with pytest.raises(ValueError):getattr(packet,key).setflags(write=True)
        old_guard();guard();save(tmp_path/(label+'.json'),packet)


@pytest.mark.parametrize('macros',(1,2))
def test_loaded_capture(macros,tmp_path):
    m,_,masses=connected(True,macros,clamped=True)
    pattern=DistributedPattern(LinePattern(tuple((eid,.005,-.003,.002) for eid in m.mesh.elements)),())
    p=Program((.5,1.),pattern);result=solve(m,p)
    assert result.status=='completed',result.failure
    save(tmp_path/'checkpoint.json',result.checkpoint)
    before=canonical(result.state)
    packet,guard=capture(m,p,result.checkpoint,masses)
    other,check=capture(m,p,result.checkpoint,masses)
    assert canonical(packet)==canonical(other) and canonical(result.state)==before
    assert np.linalg.norm(packet.net_residual[list(packet.free_dofs)])<=1e-11
    assert len(packet.internal_layout)==macros and len(packet.algebraic_dofs)==6*macros
    assert np.any(packet.geometric) and packet.history_unchanged
    if macros==1:
        supplied,modes=modal.solve_modes(m,p,result.checkpoint,masses,
            expected_sha256=sha256(result.checkpoint).hexdigest(),bounds=(0.,1e5),num_modes=6)
        assert canonical(supplied)==canonical(packet) and np.all(modes.eigenvalues>0.)
        save(tmp_path/'modes.json',modes)
    guard();check();save(tmp_path/'packet.json',packet)


@pytest.mark.parametrize('mutation',('hash','map','inertia','couple','cancel','caller','program','packet','model'))
def test_guards(mutation,tmp_path):
    m,_,masses=make(100.,True,np.eye(3));p=Program((0.,),EMPTY)
    c=Context(m,p);raw=c.checkpoint(())
    if mutation=='hash':
        with pytest.raises(ValueError,match='authority'):
            modal.prepare(m,p,raw,masses,expected_sha256='0'*64)
    elif mutation=='map':
        with pytest.raises(ValueError,match='inertia map'):capture(m,p,raw,{})
    elif mutation=='inertia':
        with pytest.raises(np.linalg.LinAlgError):capture(m,p,raw,{1:np.zeros((6,6))})
    elif mutation=='couple':
        p=Program((0.,),DistributedPattern(LinePattern(()),((1,.1,0.,0.),)))
        with pytest.raises(ValueError,match='nonconservative'):capture(m,p,raw,masses)
    elif mutation=='cancel':
        token=CancellationToken();token.cancel()
        with pytest.raises(SolveCancelled):capture(m,p,raw,masses,cancellation_token=token)
    else:
        packet,guard=capture(m,p,raw,masses)
        if mutation=='caller':masses[1][0,0]*=2
        elif mutation=='program':object.__setattr__(p,'targets',(1.,))
        elif mutation=='packet':object.__setattr__(packet,'identity','0'*64)
        else:m.materials.clear()
        with pytest.raises(ValueError):guard()
    save(tmp_path/'guard.json',dict(mutation=mutation,rejected=True))


@pytest.mark.parametrize('mutation',('branch','yield','history'))
def test_material_fail_closed(mutation,tmp_path):
    m,_,_=make(100.,False,np.eye(3));c=Context(m,Program((0.,),EMPTY))
    _,_,_,responses,_=c.assemble(c.initial.mechanical,0.,c.initial.histories)
    response=responses[0];core=m.mesh.elements[1].operator
    material=json.loads(response.material)
    history=response.history
    if mutation=='branch':material['stations'][0]['branch']='PLASTIC'
    elif mutation=='yield':material['stations'][0]['resultants']=[[core.section.yield_force,0.,0.,0.,0.,0.],[0.]*6]
    else:history=replace(history,cell_identity='changed')
    changed=SimpleNamespace(material=canonical(material),history=history)
    with pytest.raises(ValueError):modal._elastic_interior(core,changed,c.initial.histories[0])
    save(tmp_path/'material.json',dict(mutation=mutation,rejected=True))
