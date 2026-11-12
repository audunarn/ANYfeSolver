"""Static interface identities and fixed-origin state safety, not qualification."""
from dataclasses import asdict
from decimal import localcontext
import numpy as np
import pytest
from anyfesolver._ge_beam3_fibre_static_boundary import solve_static
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_fibre_line_program import model
from test_ge_beam3_schur_line_program import save


def inputs(curved,plastic):
    op=model(curved=curved,plastic=plastic).mesh.elements[1].operator
    d=np.array([[0.,0.,0.],[.014,.002,-.001],[.035,.004,-.002]])
    x=op.reference.coordinates+d
    q=np.array([rotation(v)@r for v,r in zip(([0.,0.,0.],[.006,-.012,.008],[.015,-.02,.01]),op.reference.nodal_triads)])
    return op,x,np.zeros((3,3)),q


def solve(op,x,low,q,origin=None,seed=None,load=None,**kwargs):
    return solve_static(op,x,low,q,origin=op.cell.virgin() if origin is None else origin,
        initial_rotations=np.tile(np.eye(3),(2,1,1)) if seed is None else seed.rotations,
        initial_resultants=np.zeros(18) if seed is None else seed.resultants,
        spatial_line_force=np.zeros(3) if load is None else load,**kwargs)


@pytest.mark.parametrize('curved,plastic',[(False,False),(True,False),(True,True)])
def test_static_stationarity_and_schur_work(curved,plastic,tmp_path):
    op,x,low,q=inputs(curved,plastic); origin=op.cell.virgin(); before=canonical(origin)
    result=solve(op,x,low,q,origin); h=result.full_hessian
    expected=h[:18,:18]-h[:18,18:]@np.linalg.solve(h[18:,18:],h[18:,:18])
    assert result.internal_error<=1e-11
    np.testing.assert_allclose(result.hessian,expected,rtol=1e-11,atol=1e-13)
    assert np.linalg.norm(result.hessian-result.hessian.T)<=1e-11*max(1.,np.linalg.norm(result.hessian))
    direction=np.sin(np.arange(18)+.3); internal=-np.linalg.solve(h[18:,18:],h[18:,:18]@direction)
    complete=np.r_[direction,internal]
    assert abs(complete@h@complete-direction@result.hessian@direction)<=1e-11*max(1.,abs(complete@h@complete))
    assert canonical(origin)==before
    if plastic: assert any(row[2]>0 for station in result.history.stations for row in station.rows)
    assert not result.production_qualified and not result.dynamic_reduction_authorized
    save(tmp_path/'static.json',asdict(result))


@pytest.mark.parametrize('plastic',[False,True])
def test_condensed_directional_tangent_includes_line_work(plastic,tmp_path):
    op,x,low,q=inputs(True,plastic); load=np.array([.03,-.02,.01]); origin=op.cell.virgin()
    center=solve(op,x,low,q,origin,load=load)
    direction=np.cos(np.arange(18)+.2); direction/=np.linalg.norm(direction); v=direction.reshape(3,6)
    eps=1e-6; outputs=[]
    for sign in (-1,1):
        trial_q=np.array([rotation(sign*eps*row[3:])@frame for row,frame in zip(v,q)])
        outputs.append(solve(op,x+sign*eps*v[:,:3],low,trial_q,origin,seed=center,load=load))
    jac=center.hessian.copy()
    for i in range(3):
        start=6*i+3; a,b,c=center.residual[start:start+3]
        jac[start:start+3,start:start+3]-=.5*np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
    observed=(outputs[1].residual-outputs[0].residual)/(2*eps)
    error=float(np.linalg.norm(observed-jac@direction)/max(1.,np.linalg.norm(observed)))
    assert error<=1e-7
    energy=(outputs[1].potential-outputs[0].potential)/(2*eps)
    assert abs(energy-center.residual@direction)<=1e-7*max(1.,abs(energy))
    save(tmp_path/'directional.json',dict(error=error,energy_derivative=energy,force_work=float(center.residual@direction),center=asdict(center)))


def test_fixed_origin_unload_replay_and_failure_do_not_commit(tmp_path):
    op,x,low,q=inputs(True,True); virgin=op.cell.virgin()
    first=solve(op,x,low,q,virgin); origin=first.history; snapshot=canonical(origin)
    with localcontext() as context:
        context.prec=16
        second=solve(op,op.reference.coordinates+.4*(x-op.reference.coordinates),low,q,origin,seed=first)
        assert context.prec==16
    replay=op.evaluate(op.reference.coordinates+.4*(x-op.reference.coordinates),low,q,second.rotations,second.resultants,origin=origin)
    assert canonical(replay.history)==canonical(second.history) and canonical(origin)==snapshot
    with pytest.raises(ValueError,match='iteration limit'): solve(op,x,low,q,origin,max_iterations=0)
    assert canonical(origin)==snapshot
    def cancelled(): raise RuntimeError('cancelled static trial')
    with pytest.raises(RuntimeError,match='cancelled'): solve(op,x,low,q,origin,check=cancelled)
    assert canonical(origin)==snapshot
    with pytest.raises(ValueError): second.rotations.setflags(write=True)
    with pytest.raises(ValueError): solve_static(op,x,low,q,origin=None,initial_rotations=first.rotations,initial_resultants=first.resultants,spatial_line_force=np.zeros(3))
    save(tmp_path/'history.json',dict(first=asdict(first),second=asdict(second),origin_sha256=__import__('hashlib').sha256(snapshot).hexdigest()))
