"""Distributed-couple work and general static Schur identities, private gate."""
from dataclasses import asdict
from decimal import localcontext
import ast
from pathlib import Path
import numpy as np
import pytest
from anyfesolver._ge_beam3_fibre_distributed_couples import cell_couple_load, spatial_jacobian, chart_pullback, solve_distributed_static
from anyfesolver._ge_beam3_fibre_line_work import evaluate as line_work
from anyfesolver._ge_beam3_p5.algebra import rotation, skew
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases import ge_beam3_distributed_couple_oracle as oracle
from test_ge_beam3_fibre_static_boundary import inputs, solve as conservative_solve
from test_ge_beam3_schur_line_program import save

LINE=np.array([.03,-.02,.01])
COUPLE=np.array([.07,-.04,.05])


def solve(op,x,low,q,*,origin=None,seed=None,line=LINE,couple=COUPLE,**kwargs):
    return solve_distributed_static(op,x,low,q,origin=op.cell.virgin() if origin is None else origin,
        initial_rotations=np.tile(np.eye(3),(2,1,1)) if seed is None else seed.rotations,
        initial_resultants=np.zeros(18) if seed is None else seed.resultants,
        spatial_line_force=line,spatial_couple_density=couple,**kwargs)


def independent_residual(op,x,low,q,u,p,origin,line,couple):
    # Reuse only the preserved material and conservative line operators.
    e=op.evaluate(x,low,q,u,p,origin=origin)
    w=line_work(op.reference,x,low,u,line,order=op.order)
    return e.residual-w.gradient-oracle.load(op.reference.coordinates,couple,op.order)


@pytest.mark.parametrize('curved',[False,True])
def test_reference_work_and_closed_chart(curved,tmp_path):
    op,_,_,_=inputs(curved,False);actual=cell_couple_load(op,COUPLE)
    expected=oracle.load(op.reference.coordinates,COUPLE,op.order)
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-11)
    assert not np.any(actual[:18]) and not np.any(actual[24:])
    u=np.array([rotation([.4,-.2,.3]),rotation([-.3,.2,.1])]);variation=np.array([[.2,.1,-.3],[-.1,.4,.2]])
    observed=0.;eps=1e-6
    for cell,_,_,measure,frame in op.stations:
        current=u[cell]@frame
        derivative=(rotation(eps*variation[cell])@current-rotation(-eps*variation[cell])@current)/(2*eps)
        spin=derivative@current.T
        axial=np.array([spin[2,1]-spin[1,2],spin[0,2]-spin[2,0],spin[1,0]-spin[0,1]])/2
        observed+=measure*COUPLE@axial
    work_error=float(abs(observed-actual[18:24]@variation.ravel()));assert work_error<=1e-7
    force=np.sin(np.arange(18));jac=np.arange(324,dtype=float).reshape(18,18)/300
    increments=np.array([[.3,-.2,.4],[0.,0.,0.],[.002,-.001,.003]])
    pulled,tangent,_=chart_pullback(force,jac,increments);f,k=oracle.pullback(force,jac,increments)
    np.testing.assert_allclose(pulled,f,atol=1e-11,rtol=0.)
    np.testing.assert_allclose(tangent,k,atol=1e-11,rtol=0.)
    save(tmp_path/'work.json',dict(load=actual,work_error=work_error,chart_force=pulled,chart_tangent=tangent,production_qualified=False))


@pytest.mark.parametrize('curved,plastic',[(False,False),(True,False),(True,True)])
def test_spatial_internal_jacobian_and_general_schur(curved,plastic,tmp_path):
    op,x,low,q=inputs(curved,plastic);origin=op.cell.virgin();before=canonical(origin)
    center=solve(op,x,low,q,origin=origin);j=center.full_spatial_jacobian
    np.testing.assert_allclose(center.conservative_residual-center.applied_couple,center.full_residual,atol=0.,rtol=0.)
    assert center.internal_error<=1e-11 and np.linalg.norm(j[18:,18:]-j[18:,18:].T)>1e-3
    lifted=-np.linalg.solve(j[18:,18:],j[18:,:18])
    expected=j[:18,:18]+j[:18,18:]@lifted
    np.testing.assert_allclose(center.internal_lift,lifted,atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(center.spatial_jacobian,expected,atol=1e-11,rtol=1e-11)
    direction=np.sin(np.arange(42)+.3);direction/=np.linalg.norm(direction);eps=1e-6
    outputs=[]
    for sign in (-1,1):
        d=sign*eps*direction;delta=d[:18].reshape(3,6)
        nq=np.array([rotation(row[3:])@r for row,r in zip(delta,q)])
        nu=np.array([rotation(d[18+3*c:21+3*c])@center.rotations[c] for c in (0,1)])
        outputs.append(independent_residual(op,x+delta[:,:3],low,nq,nu,center.resultants+d[24:],origin,LINE,COUPLE))
    observed=(outputs[1]-outputs[0])/(2*eps)
    error=float(np.linalg.norm(observed-j@direction)/max(1.,np.linalg.norm(observed)));assert error<=1e-7
    wrong=spatial_jacobian(center.full_residual,center.conservative_hessian)
    wrong_error=float(np.linalg.norm(wrong-j));assert wrong_error>1e-3
    h=center.conservative_hessian;old=h[:18,:18]-h[:18,18:]@np.linalg.solve(h[18:,18:],h[18:,:18])
    for n in range(3):old[6*n+3:6*n+6,6*n+3:6*n+6]-=.5*skew(center.residual[6*n+3:6*n+6])
    old_error=float(np.linalg.norm(old-center.spatial_jacobian));assert old_error>1e-3
    assert canonical(origin)==before and not hasattr(center,'potential')
    assert not center.conservative_potential and not center.conservative_spectral_authority and not center.production_qualified
    save(tmp_path/'schur.json',dict(center=asdict(center),directional_error=error,wrong_net_connection_error=wrong_error,
        wrong_conservative_reduction_error=old_error,origin_unchanged=True))


@pytest.mark.parametrize('plastic',[False,True])
def test_condensed_spatial_and_additive_chart_derivatives(plastic,tmp_path):
    op,x,low,q=inputs(True,plastic);origin=op.cell.virgin();center=solve(op,x,low,q,origin=origin)
    direction=np.cos(np.arange(18)+.2);direction/=np.linalg.norm(direction);v=direction.reshape(3,6);eps=1e-6
    spatial=[];chart=[];increments=np.array([[0.,0.,0.],[.006,-.012,.008],[.015,-.02,.01]])
    for sign in (-1,1):
        nq=np.array([rotation(sign*eps*row[3:])@r for row,r in zip(v,q)])
        spatial.append(solve(op,x+sign*eps*v[:,:3],low,nq,origin=origin,seed=center).residual)
        inc=increments+sign*eps*v[:,3:]
        nq=np.array([rotation(row)@r for row,r in zip(inc,op.reference.nodal_triads)])
        r=solve(op,x+sign*eps*v[:,:3],low,nq,origin=origin,seed=center)
        # Observe the chart force through the separate closed Rodrigues map.
        chart.append(oracle.pullback(r.residual,r.spatial_jacobian,inc)[0])
    observed=(spatial[1]-spatial[0])/(2*eps)
    spatial_error=float(np.linalg.norm(observed-center.spatial_jacobian@direction)/max(1.,np.linalg.norm(observed)))
    _,tangent,_=chart_pullback(center.residual,center.spatial_jacobian,increments)
    observed=(chart[1]-chart[0])/(2*eps)
    chart_error=float(np.linalg.norm(observed-tangent@direction)/max(1.,np.linalg.norm(observed)))
    assert max(spatial_error,chart_error)<=1e-7
    assert np.linalg.norm(tangent-tangent.T)>1e-3
    save(tmp_path/'directional.json',dict(spatial_error=spatial_error,chart_error=chart_error,chart_tangent=tangent,production_qualified=False))


@pytest.mark.parametrize('curved,plastic',[(False,False),(True,False),(True,True)])
def test_zero_couple_recovers_preserved_conservative_state(curved,plastic,tmp_path):
    op,x,low,q=inputs(curved,plastic);origin=op.cell.virgin()
    old=conservative_solve(op,x,low,q,origin,load=LINE)
    new=solve(op,x,low,q,origin=origin,couple=np.zeros(3))
    for key in ('rotations','resultants','residual','history'):
        assert canonical(getattr(old,key))==canonical(getattr(new,key))
    assert canonical(old.full_hessian)==canonical(new.conservative_hessian)
    expected=old.hessian.copy()
    for n in range(3):expected[6*n+3:6*n+6,6*n+3:6*n+6]-=.5*skew(old.residual[6*n+3:6*n+6])
    error=float(np.linalg.norm(expected-new.spatial_jacobian)/max(1.,np.linalg.norm(expected)))
    assert error<=1e-11
    save(tmp_path/'zero.json',dict(physical_state_byte_identical=True,jacobian_error=error,production_qualified=False))


def test_large_rigid_superposition_and_fixed_origin_unload(tmp_path):
    op,x,low,q=inputs(True,True);origin=op.cell.virgin();a=solve(op,x,low,q,origin=origin)
    common=rotation([2.5,-.7,.8]);shift=np.array([.25,-.125,.0625])
    b=solve_distributed_static(op,x@common.T+shift,low@common.T,common@q,origin=origin,
        initial_rotations=common@a.rotations,initial_resultants=a.resultants,
        spatial_line_force=common@LINE,spatial_couple_density=common@COUPLE)
    transform=np.kron(np.eye(6),common)
    r_error=float(np.linalg.norm(b.residual-transform@a.residual)/max(1.,np.linalg.norm(a.residual)))
    j_error=float(np.linalg.norm(b.spatial_jacobian-transform@a.spatial_jacobian@transform.T)/max(1.,np.linalg.norm(a.spatial_jacobian)))
    assert max(r_error,j_error)<=1e-11
    origins=a.history;before=canonical(origins)
    with localcontext() as ctx:
        ctx.prec=16
        unloaded=solve(op,op.reference.coordinates+.4*(x-op.reference.coordinates),low,q,origin=origins,seed=a,line=.4*LINE,couple=-.4*COUPLE)
        assert ctx.prec==16
    replay=op.evaluate(op.reference.coordinates+.4*(x-op.reference.coordinates),low,q,unloaded.rotations,unloaded.resultants,origin=origins)
    assert canonical(replay.history)==canonical(unloaded.history) and canonical(origins)==before
    with pytest.raises(ValueError,match='iteration limit'):solve(op,x,low,q,origin=origins,max_iterations=0)
    def cancelled():raise RuntimeError('cancelled internal trial')
    with pytest.raises(RuntimeError,match='cancelled'):solve(op,x,low,q,origin=origins,check=cancelled)
    assert canonical(origins)==before
    with pytest.raises(ValueError):unloaded.spatial_jacobian.setflags(write=True)
    save(tmp_path/'objectivity-history.json',dict(residual_error=r_error,jacobian_error=j_error,unloaded=asdict(unloaded),origin_unchanged=True))


@pytest.mark.parametrize('bad',[None,[0.,0.],[float('nan'),0.,0.],[float('inf'),0.,0.],[1e200,0.,0.]])
def test_bad_density_rejected(bad):
    op,_,_,_=inputs(False,False)
    with pytest.raises(ValueError):cell_couple_load(op,bad)


def test_deadline_and_source_separation(monkeypatch):
    import anyfesolver._ge_beam3_fibre_distributed_couples as module
    op,x,low,q=inputs(False,False);times=[]
    def clock():times.append(True);return 61.*len(times)
    monkeypatch.setattr(module,'monotonic',clock)
    with pytest.raises(RuntimeError,match='deadline'):solve(op,x,low,q)
    imports=[]
    for node in ast.walk(ast.parse(Path(oracle.__file__).read_text())):
        if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert set(imports)=={'math','numpy'}
