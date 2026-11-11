"""Constrained generalized-cell and retained finite-rotation development checks."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_generalized_cell import GeneralizedCellConjugate
from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from anyfesolver._ge_beam3_fibre_section import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from docs.reference_cases.ge_beam3_generalized_cell_audit import audit
from test_ge_beam3_generalized_ellipsoid_section import section
from test_ge_beam3_fibre_cell import stations
from test_ge_beam3_curved_contrast_probe import make_curved
from test_ge_beam3_retained_plastic import args
from test_ge_beam3_schur_line_program import save


def inputs(order=2):
    law=section();data=stations(order);return law,data,GeneralizedCellConjugate(law,data)


def check(law,data,p,r):
    description=dict(elastic=law.elastic.tolist(),metric=law.metric.tolist(),yield_force=law.yield_force,hardening=law.hardening)
    return audit(description,data,p.tolist(),json.loads(canonical(r)))


def vector(pair):return np.array(pair[0])+np.array(pair[1])


@pytest.mark.parametrize('order',[2,4,8])
def test_station_primal_dual_work_and_independent_tangent(order,tmp_path):
    law,data,cell=inputs(order);p=np.linspace(-.3,.5,18);r=cell.response(p.tolist());errors=check(law,data,p,r)
    save(tmp_path/'cell.json',dict(response=r,errors=errors))
    assert max(errors.values())<=1e-11
    assert len(r['stations'])==2*order
    assert all(any(sum(history.plastic[i])!=0 for history in r['history'].stations) for i in range(6))
    assert canonical(r)==canonical(cell.response(p.tolist()))


def test_cell_history_and_directional_derivatives(tmp_path):
    law,data,cell=inputs();p=np.linspace(-.3,.5,18);origin=cell.virgin();rows=[]
    for factor in (1.,.8,-1.,.5):
        old=canonical(origin);r=cell.response((factor*p).tolist(),origin);errors=check(law,data,factor*p,r)
        assert max(errors.values())<=1e-11 and canonical(origin)==old
        rows.append(dict(response=r,errors=errors));origin=r['history']
    d=.1*np.cos(np.arange(18));h=1e-5;r=cell.response(p.tolist(),origin)
    plus=cell.response((p+h*d).tolist(),origin);minus=cell.response((p-h*d).tolist(),origin)
    gradient=vector(r['gradient']);tangent=np.array([vector(row) for row in r['hessian']])
    first=abs((vector(plus['potential'])[0]-vector(minus['potential'])[0])/(2*h)-gradient@d)
    second=float(np.linalg.norm((vector(plus['gradient'])-vector(minus['gradient']))/(2*h)-tangent@d)/max(1.,np.linalg.norm(tangent@d)))
    assert first<=1e-7 and second<=1e-7
    save(tmp_path/'history.json',dict(rows=rows,first=first,second=second))


@pytest.mark.parametrize('mutation',['force','moment','strain','history','potential','hessian'])
def test_independent_audit_rejects_mutation(mutation,tmp_path):
    law,data,cell=inputs();p=np.linspace(-.3,.5,18);r=deepcopy(cell.response(p.tolist()))
    if mutation=='force':r['stations'][0]['resultants'][0][0]+=1.
    elif mutation=='moment':r['stations'][0]['resultants'][0][3]+=1.
    elif mutation=='strain':r['stations'][0]['strain'][0][0]+=1.
    elif mutation=='potential':r['potential'][0][0]+=1.
    elif mutation=='hessian':r['hessian'][0][0][0]+=1.
    else:
        rows=list(r['history'].stations);rows[0]=replace(rows[0],accumulated=(100.,0.));r['history']=replace(r['history'],stations=tuple(rows))
    errors=check(law,data,p,r);assert max(errors.values())>1e-11
    save(tmp_path/'mutation.json',dict(mutation=mutation,errors=errors,rejected=True))


def test_cancellation_and_authority_fail_closed(monkeypatch,tmp_path):
    law,data,cell=inputs();origin=cell.virgin();saved=canonical(origin)
    def cancel():raise RuntimeError('cancelled fixture')
    with pytest.raises(RuntimeError,match='cancelled'):cell.response([0.]*18,origin,check=cancel)
    with pytest.raises(ValueError):cell.response([0.]*18,replace(origin,stations=origin.stations[:-1]))
    with pytest.raises(ValueError):cell.response([0.]*18,replace(origin,cell_identity='0'*64))
    changed=deepcopy(data);changed[0]['weight']=0.
    with pytest.raises(ValueError):GeneralizedCellConjugate(law,changed)
    import anyfesolver._ge_beam3_generalized_cell as module
    ticks=[]
    def clock():ticks.append(1);return 61.*len(ticks)
    monkeypatch.setattr(module,'monotonic',clock)
    with pytest.raises(TimeoutError):cell.response([0.]*18,origin)
    assert canonical(origin)==saved
    save(tmp_path/'rejected.json',dict(cancelled=True,deadline=True,foreign_history=True,bad_station=True))


@pytest.mark.parametrize('kind',['straight','curved'])
def test_retained_variations_and_objectivity(kind,tmp_path):
    ref=make_curved(100.)[0].mesh.elements[1].core.reference
    if kind=='straight':
        from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
        ref=CenteredCurvedBeam3ReferenceGeometry(np.array([[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]]),np.array([np.eye(3)]*3))
    probe=RetainedGeneralizedOperator(ref,section(),order=4);state=args(probe)
    origin=probe.cell.response((state[-1]*.5).tolist())['history'];saved=canonical(origin)
    d=.1*np.cos(np.arange(42));centre=.01*np.sin(np.arange(42));eps=1e-5
    r=probe.evaluate(*state,origin=origin,increment=centre)
    plus=probe.evaluate(*state,origin=origin,increment=centre+eps*d);minus=probe.evaluate(*state,origin=origin,increment=centre-eps*d)
    tangent=r.hessian+r.hessian_low
    first=abs((plus.potential-minus.potential)/(2*eps)-r.residual@d)
    second=float(np.linalg.norm((plus.residual-minus.residual)/(2*eps)-tangent@d)/max(1.,np.linalg.norm(tangent@d)))
    symmetry=float(np.linalg.norm(tangent-tangent.T)/max(1.,np.linalg.norm(tangent)))
    assert first<=1e-7 and second<=1e-7 and symmetry<=1e-11
    x,low,q,u,p=state;common=rotation([2.6,.8,-.3]);shift=[1000.,-2000.,3000.]
    moved=np.zeros_like(x);moved_low=np.zeros_like(low)
    for n in range(3):
        for i in range(3):moved[n,i],moved_low[n,i]=split_sum([shift[i],*(float(common[i,j]*x[n,j]) for j in range(3))])
    a=probe.evaluate(*state,origin=origin);b=probe.evaluate(moved,moved_low,common@q,common@u,p,origin=origin)
    transform=np.eye(42);transform[:24,:24]=np.kron(np.eye(8),common)
    objective=dict(energy=abs(a.potential-b.potential),residual=float(np.linalg.norm(transform@a.residual-b.residual)),
        tangent=float(np.linalg.norm(transform@a.hessian@transform.T-b.hessian)/max(1.,np.linalg.norm(a.hessian))))
    assert max(objective.values())<=1e-11 and a.material==b.material and canonical(origin)==saved
    before=probe.recover(u,p,origin=origin);after=probe.recover(common@u,p,origin=origin)
    for row,other in zip(before,after):
        assert 'fibres' not in row
        for key in ('strain','strain_low','resultants','resultants_low','history'):assert canonical(row[key])==canonical(other[key])
        np.testing.assert_allclose(common@row['current_frame'],other['current_frame'],atol=1e-11,rtol=0.)
    save(tmp_path/'retained.json',dict(kind=kind,first=first,second=second,symmetry=symmetry,objectivity=objective,recovery=before))



def test_cell_proper_spatial_reexpression(tmp_path):
    law,data,cell=inputs();p=np.linspace(-.3,.5,18);a=cell.response(p.tolist())
    r=rotation([.4,-.3,.2]);other=deepcopy(data)
    for row in other:row['v']=(np.array(row['v'])@r.T).tolist()
    rotated=GeneralizedCellConjugate(law,other);rp=p.copy();rp[:6]=(p[:6].reshape(2,3)@r.T).ravel()
    b=rotated.response(rp.tolist());transform=np.eye(18);transform[:6,:6]=np.kron(np.eye(2),r)
    ga=vector(a['gradient']);ha=np.array([vector(row) for row in a['hessian']])
    gb=vector(b['gradient']);hb=np.array([vector(row) for row in b['hessian']])
    errors=dict(gradient=float(np.linalg.norm(transform@ga-gb)),hessian=float(np.linalg.norm(transform@ha@transform.T-hb)/max(1.,np.linalg.norm(ha))))
    assert max(errors.values())<=1e-11
    for aa,bb in zip(a['stations'],b['stations']):
        np.testing.assert_allclose(vector(aa['strain']),vector(bb['strain']),atol=1e-11,rtol=0.)
        np.testing.assert_allclose(vector(aa['resultants']),vector(bb['resultants']),atol=1e-11,rtol=0.)
    save(tmp_path/'covariance.json',dict(errors=errors,base=a,rotated=b))


@pytest.mark.parametrize('kind',['straight','curved'])
def test_reference_condensation_rigid_nulls_and_positive_complement(kind,tmp_path):
    ref=make_curved(100.)[0].mesh.elements[1].core.reference
    if kind=='straight':
        from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
        ref=CenteredCurvedBeam3ReferenceGeometry(np.array([[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]]),np.array([np.eye(3)]*3))
    probe=RetainedGeneralizedOperator(ref,section(),order=4)
    result=probe.evaluate(ref.coordinates,np.zeros((3,3)),ref.nodal_triads,np.array([np.eye(3)]*2),np.zeros(18))
    h=result.hessian+result.hessian_low
    lift=-np.linalg.solve(h[18:,18:],h[18:,:18]);k=h[:18,:18]+h[:18,18:]@lift
    rigid=np.zeros((18,6))
    for node,x in enumerate(ref.coordinates):
        rigid[6*node:6*node+3,:3]=np.eye(3);rigid[6*node+3:6*node+6,3:]=np.eye(3)
        for j in range(3):rigid[6*node:6*node+3,3+j]=np.cross(np.eye(3)[j],x)
    orthogonal=np.linalg.qr(rigid,mode='complete')[0];z=orthogonal[:,6:]
    error=float(np.linalg.norm(k@rigid)/max(1.,np.linalg.norm(k)*np.linalg.norm(rigid)))
    symmetry=float(np.linalg.norm(k-k.T)/max(1.,np.linalg.norm(k)))
    eigen=np.linalg.eigvalsh(z.T@k@z)
    assert np.linalg.matrix_rank(rigid)==6 and error<=1e-11 and symmetry<=1e-11
    assert len(eigen)==12 and min(eigen)>0 and np.linalg.norm(result.residual)<=1e-11
    save(tmp_path/'reference.json',dict(kind=kind,rigid_error=error,symmetry=symmetry,positive_complement_eigenvalues=eigen,
        condensed=k,rigid=rigid,lift=lift,numerical_reference_check_only=True,production_qualified=False))
