"""Small-cut numerical-domain development checks; not D1 acceptance."""
import hashlib,json,math,sys
from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_enriched import EnrichedProfileError
from anyfesolver.general_contact.enriched_overlap import QuadraturePolicy,EnrichedGeometryError
from anyfesolver.general_contact import enriched_small_cut as smallcut
from anyfesolver.general_contact import enriched_moment_batch as kernel
from anyfesolver.general_contact import enriched_union as scalar
from anyfesolver.general_contact.moment_batch_ad import PointJet

POLICY=QuadraturePolicy(1e-10,1e-8,1e-5,14144)

def rule25():
    x,w=np.polynomial.legendre.leggauss(5);rule=[]
    for a,wa in zip(x,w):
        a=(a+1)/2
        for b,wb in zip(x,w):
            b=(b+1)/2
            rule.append(((1-a-(1-a)*b,a,(1-a)*b),wa*wb*(1-a)/2))
    return rule

def constant(x):return Jet2.constant(x,37)
def p_add(a,b):
    made=dict(a)
    for key,value in b.items():made[key]=made.get(key,0)+value
    return made
def p_scale(a,b):return {k:v*b for k,v in a.items()}
def p_mul(a,b):
    made={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():made[i+k,j+l]=made.get((i+k,j+l),0)+x*y
    return made
def p_sub(a,b):return p_add(a,p_scale(b,-1))
def p_sum(values):
    out={}
    for value in values:out=p_add(out,value)
    return out

def polynomial_triangle(anchor=0.):
    """Two formal scalar variables span every37coordinate; all Hessian entries tested."""
    s=2.**-24;a=np.arange(1.,38.)/64;b=(-1.)**np.arange(37)/128
    aa=Jet2(0.,a,np.zeros((37,37)));bb=Jet2(0.,b,np.zeros((37,37)))
    vertices=((anchor+aa,anchor+bb),(anchor+s+2*aa+bb,anchor+aa+3*bb),(anchor-aa+bb,anchor+s+2*bb))
    x=({(1,0):1},{(0,0):s,(1,0):2,(0,1):1},{(1,0):-1,(0,1):1})
    y=({(0,1):1},{(1,0):1,(0,1):3},{(0,0):s,(0,1):2})
    x=tuple(p_add(v,{(0,0):anchor})for v in x);y=tuple(p_add(v,{(0,0):anchor})for v in y)
    area=p_scale(p_sub(p_mul(p_sub(x[1],x[0]),p_sub(y[2],y[0])),p_mul(p_sub(y[1],y[0]),p_sub(x[2],x[0]))),.5)
    # Independent uniform-simplex moments E(lambda_i^2)=1/6,
    # E(lambda_i lambda_j)=1/12; physical area, not2*area.
    sx,sy=p_sum(x),p_sum(y)
    xx=p_scale(p_add(p_sum(p_mul(v,v)for v in x),p_sum(p_mul(x[i],x[j])for i in range(3)for j in range(i+1,3))),1/6)
    yy=p_scale(p_add(p_sum(p_mul(v,v)for v in y),p_sum(p_mul(y[i],y[j])for i in range(3)for j in range(i+1,3))),1/6)
    xy=p_scale(p_add(p_sum(p_mul(x[i],y[i])for i in range(3)),p_mul(sx,sy)),1/12)
    polynomials=[area,p_mul(area,p_scale(sx,1/3)),p_mul(area,p_scale(sy,1/3)),p_mul(area,xx),p_mul(area,xy),p_mul(area,yy)]
    def coefficients(poly):
        first=poly.get((1,0),0)*a+poly.get((0,1),0)*b
        second=(2*poly.get((2,0),0)*np.outer(a,a)+poly.get((1,1),0)*(np.outer(a,b)+np.outer(b,a))+2*poly.get((0,2),0)*np.outer(b,b))
        return poly.get((0,0),0),first,second
    return vertices,polynomials,coefficients

def polynomial_integrand(points):
    x=PointJet.stack([p[0]for p in points]);y=PointJet.stack([p[1]for p in points])
    one=PointJet.constant(1.,25,37)
    return [(k+1)*v for k in range(4)for v in (one,x,y,x*x,x*y,y*y)]

def test_all24_simplex_polynomials_preserve_full37_area_chain_and_normalization(record_property):
    reports=[]
    for anchor in (0.,.25):
        vertices,polynomials,coefficients=polynomial_triangle(anchor)
        out,triangles,difference,bounds=smallcut.chart_integral(vertices,polynomial_integrand,rule25(),np.array((1e-10,1e-8,1e-5)),14144)
        assert len(out)==24 and triangles>=5
        for k,observed in enumerate(out):
            value,g,h=coefficients(p_scale(polynomials[k%6],k//6+1))
            np.testing.assert_allclose(observed.value,value,rtol=1e-11,atol=1e-32)
            np.testing.assert_allclose(observed.gradient,g,rtol=1e-11,atol=1e-28)
            np.testing.assert_allclose(observed.hessian,h,rtol=1e-11,atol=1e-24)
        assert np.count_nonzero(out[0].gradient)>0 and np.count_nonzero(out[0].hessian)>0
        reports.append(dict(anchor=anchor,triangles=triangles,difference=difference.tolist(),physical_area=out[0].value,conversion_envelopes=bounds))
    # Independent protected scalar quotient includes nonzero input Hessians.
    a=np.arange(1.,38.)/64;b=(-1.)**np.arange(37)/128
    numerator=Jet2(2.,a,np.outer(a,b)+np.outer(b,a));denominator=Jet2(1.5,b,np.outer(a,a))
    exact=smallcut._Exact37.from_jet(numerator)/smallcut._Exact37.from_jet(denominator)
    conversion=smallcut._ConversionBounds();made=conversion.convert(exact,0);old=numerator/denominator
    np.testing.assert_allclose(made.value,old.value,rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(made.gradient,old.gradient,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(made.hessian,old.hessian,rtol=1e-12,atol=1e-14)
    tiny=smallcut._Exact37(Fraction(1,2**1075),tuple(Fraction(0)for _ in range(37)),tuple(tuple(Fraction(0)for _ in range(37))for _ in range(37)))
    subnormal_conversion=smallcut._ConversionBounds();rounded=subnormal_conversion.convert(tiny,0)
    assert rounded.value==0. and subnormal_conversion.report()['max_absolute_error_power2'][0]==-1075
    assert subnormal_conversion.report()['inexact_counts']==[1,0,0]
    record_property('all24_full37_analytic',json.dumps(reports))

def _triangle_context():
    # Exact non-tie represented INPUT clipping; output order is the actual
    # Sutherland-Hodgman order. Constant jets suffice for topology refusals.
    source=[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]
    s=2.**-24;target=[(.1,.1),(.1+s,.1),(.1,.1+s)]
    exact,signature=smallcut._exact_clip([tuple(Fraction(x)for x in p)for p in source],[tuple(Fraction(x)for x in p)for p in target])
    q=[[constant(x)for x in p]for p in source];t=[[constant(x)for x in p]for p in target]
    rounded,round_signature=smallcut._rounded_clip(source,target)
    assert round_signature==signature
    polygon=[[constant(float(x))for x in p]for p in rounded]
    vertices=tuple(polygon[:3])
    return vertices,q,t,polygon,signature

def test_input_topology_rounding_reconstruction_and_cancellation_refuse_without_omission():
    vertices,q,t,polygon,signature=_triangle_context()
    cert=smallcut.certify_positive_cut(vertices,q,t,polygon,signature,0)
    assert cert.rounding_area_lower>0
    with pytest.raises(EnrichedGeometryError):
        smallcut.certify_positive_cut(vertices,q,t,polygon,tuple(1-v for v in signature),0)
    negative=(vertices[0],vertices[2],vertices[1])
    with pytest.raises(EnrichedGeometryError):smallcut.chart_integral(negative,polynomial_integrand,rule25(),np.array((1e-10,1e-8,1e-5)),14144)
    ambiguous=list(q);ambiguous[0]=t[0]
    with pytest.raises(EnrichedGeometryError):smallcut.certify_positive_cut(vertices,ambiguous,t,polygon,signature,0)
    # Distinct exact local points that collapse at a huge global anchor MUST
    # refuse before invoking mechanics, rather than coarse/fine false agreement.
    close=((constant(1.),constant(1.)),(constant(np.nextafter(1.,2.)),constant(1.)),(constant(1.),constant(np.nextafter(1.,2.))))
    reached=[]
    with pytest.raises(EnrichedGeometryError,match='reconstruction'):
        smallcut.chart_integral(close,lambda p:reached.append(True),rule25(),np.array((1e-10,1e-8,1e-5)),14144)
    assert not reached
    with pytest.raises(SolveCancelled):smallcut.integrate_positive_cut(vertices,q,t,polygon,signature,0,polynomial_integrand,rule25(),np.array((1e-10,1e-8,1e-5)),14144,cancel=lambda:True)
    with pytest.raises(EnrichedGeometryError,match='budget'):
        smallcut.chart_integral(vertices,polynomial_integrand,rule25(),np.array((1e-10,1e-8,1e-5)),0)
    # Independent cut: source corner(1,1), moving line x+y=2-e+A+B.
    # Its intersections are exactly(1-e+A+B,1) and(1,1-e+A+B),
    # hence their FULL37 Hessians are zero despite intermediate quotients.
    e=2.**-24;a=np.arange(1.,38.)/64;b=(-1.)**np.arange(37)/128
    aa=Jet2(0.,a,np.zeros((37,37)));bb=Jet2(0.,b,np.zeros((37,37)))
    source=[[constant(x),constant(y)]for x,y in ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))]
    target=((1-2*e+aa,1+e+bb),(1+e+aa,1-2*e+bb),(constant(2.),constant(2.)))
    clipped,signature=smallcut._exact_clip37(source,target,lambda:False)
    assert len(clipped)==3
    expected={(Fraction(1),Fraction(1)):(None,None),
              (Fraction(1-e),Fraction(1)):(a+b,None),
              (Fraction(1),Fraction(1-e)):(None,a+b)}
    assert {tuple(x.value for x in p)for p in clipped}==set(expected)
    for point in clipped:
        for coordinate,gradient in zip(point,expected[tuple(x.value for x in point)]):
            wanted=np.zeros(37)if gradient is None else gradient
            assert coordinate.gradient==tuple(Fraction(float(v))for v in wanted)
            assert all(v==0 for row in coordinate.hessian for v in row)

def _small():
    from anyfesolver import FEModel,BoundaryCondition,create_shell_element
    from anyfesolver.general_contact.enriched_nonlinear import NonlinearEnrichedSystem,PairDefinition
    model=FEModel('small_cut_ordinary_C0_oracle');model.add_material('elastic',1e5,.3,density=1.)
    source=((-0.5,-0.5,0.),(.5,-.5,0.),(.5,.5,0.),(-.5,.5,0.))
    target=((-2.,-2.,-.2),(2.,-2.,-.2),(2.,2.,-.2),(-2.,2.,-.2))
    for node,xyz in enumerate((*source,*target),1):model.add_node(node,*xyz)
    model.add_element(1,create_shell_element(1,(1,2,3,4),'elastic',thickness=.05,reference_normal=(0.,0.,1.)))
    pairs=[]
    for element,nodes in enumerate(((5,6,8),(6,7,8)),2):
        model.add_element(element,create_shell_element(element,nodes,'elastic',thickness=.05,formulation='e4-pl-s3-v2d',reference_normal=(0.,0.,1.),reference_surface_offset=.01))
        offset=model.mesh.elements[element].material_surface_offset_from_reference(1.)
        pairs.append(PairDefinition(1,element,-.025,offset))
    model.add_boundary_condition(BoundaryCondition('fixed_nodal',tuple(model.mesh.nodes),{n:0. for n in('ux','uy','uz','rx','ry','rz')}))
    return model,NonlinearEnrichedSystem.from_model(model,pairs,normal_stiffness=1e4,quadrature=POLICY)

def test_ordinary_certified_C0_path_matches_all24_protected_scalar_jets(monkeypatch,record_property):
    model,system=_small();u=np.zeros(system.offset.size)
    for node in (5,6,7,8):
        dofs=model.mesh.dof_manager.get_node_dofs(node);u[dofs[2]]=.02*model.mesh.nodes[node].x
    pair=system.pairs[0];x=pair.map(u)[0];cert=system.union_certificates(u)[str(pair.definition.source_element)]
    args=(pair.reference_source,x[:12].reshape(4,3),x[12:21].reshape(3,3),x[21:25],pair.reference_directors[0],POLICY)
    kw=dict(current_midsurface=x[25:].reshape(4,3),internal_edge_nodes=cert['shared_nodes'],target_node_ids=pair.target_node_ids)
    observed={}
    def observe(module,label):
        constructor=module.EnrichedMoments
        def capture(*a,**k):
            frame=sys._getframe(1)
            assert frame.f_code.co_name in('selected_pair_moments','selected_batch_moments')
            jets=frame.f_locals['out'];assert len(jets)==24
            observed[label]=[(v.value,v.gradient.copy(),v.hessian.copy())for v in jets]
            return constructor(*a,**k)
        monkeypatch.setattr(module,'EnrichedMoments',capture)
    observe(scalar,'old');observe(kernel,'new')
    def forbidden(*a,**k):raise AssertionError('ordinary admitted internal tie must retain original path')
    monkeypatch.setattr(smallcut,'integrate_positive_cut',forbidden)
    old=scalar.selected_pair_moments(*args,**kw);new=kernel.selected_batch_moments(*args,**kw)
    errors=[0.,0.,0.]
    for a,b in zip(observed['new'],observed['old']):
        for i,tol in enumerate((1e-10,1e-8,1e-5)):
            errors[i]=max(errors[i],float(np.max(np.abs(np.asarray(a[i])-np.asarray(b[i])))))
            assert errors[i]<=tol
    assert old.branch==new.branch
    record_property('ordinary_all24_errors',json.dumps(dict(errors=errors,old_triangles=old.triangles,new_triangles=new.triangles,old_difference=old.quadrature_difference,new_difference=new.quadrature_difference)))

def _saved():
    path=Path(__file__).parent/'data/enriched_small_cut_v13.json';data=json.loads(path.read_bytes())
    # Keep the original absolute provenance in the frozen case record, but load
    # its exact byte copy from the checkout on every host.
    provenance=path.with_name('enriched_small_cut_v13_provenance.json')
    assert hashlib.sha256(provenance.read_bytes()).hexdigest()==data['provenance_sha256']
    return data,np.array([float.fromhex(x)for x in data['coordinates_hex']]),np.array([float.fromhex(x)for x in data['forces_hex']])

def test_saved_physical_cut_reaches_actual_caller_and_retains_all_modes(monkeypatch,record_property):
    from examples.general_contact_enriched_matched import build
    _,system,_,_=build();data,u,z=_saved();completed=[];actual=smallcut.integrate_positive_cut
    def observed(*args,**kw):
        result=actual(*args,**kw);completed.append(result[3]);return result
    monkeypatch.setattr(smallcut,'integrate_positive_cut',observed)
    out=system.linearize(u,z)
    assert completed and len(out.support)==25 and np.all(out.support>0)
    assert out.gap_gradient.shape==(25,238)and np.all(np.isfinite(out.tangent))
    assert system._accepted is None and not system._busy
    record_property('saved_cut_certificates',json.dumps(completed))
    record_property('saved_capture_sha256',data['provenance_sha256'])

def test_actual_owner_cancellation_mutation_and_accepted_checkpoint_are_atomic(monkeypatch):
    from anyfesolver.general_contact.enriched_nonlinear import MixedCriteria,IterationControl
    model,system=_small();system.solve(np.zeros(system.offset.size),criteria=MixedCriteria(.04,1e-6,1e-8),control=IterationControl(1,1))
    accepted=system._accepted;checkpoint=system.checkpoint();entered=[];actual=kernel.selected_batch_moments
    def cancelled(*a,**kw):
        entered.append(True);kw['cancel']=lambda:True;return actual(*a,**kw)
    with monkeypatch.context()as p:
        p.setattr(kernel,'selected_batch_moments',cancelled)
        with pytest.raises(SolveCancelled):system.linearize(np.zeros(system.offset.size),np.zeros(4))
    assert entered and system._accepted is accepted and system.checkpoint()==checkpoint and not system._busy
    old=model.mesh.nodes[1].z;model.mesh.nodes[1].z=old+.001;reached=[]
    try:
        with monkeypatch.context()as p:
            p.setattr(smallcut,'integrate_positive_cut',lambda *a,**kw:reached.append(True))
            with pytest.raises(EnrichedProfileError,match='definition changed'):system.linearize(np.zeros(system.offset.size),np.zeros(4))
        assert not reached and system._accepted is accepted
    finally:model.mesh.nodes[1].z=old
    assert system.checkpoint()==checkpoint and not system._busy
