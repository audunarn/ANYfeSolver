"""Bounded full-coverage proof for fixed-plane affine S3 friction paths.

Exact rational polynomial coefficients enclose fixed-map evaluation roundoff.
This is a sufficient condition, not a sampled-position certificate. Curved,
varying-normal and unresolved paths are refused; no ordinary fallback admits
them for the new physical-side/predictor friction combination.
"""
from fractions import Fraction as F
from math import comb
import numpy as np


def add(a,b):return tuple((a[i][0]+b[i][0],a[i][1]+b[i][1]) for i in range(len(a)))
def neg(a):return tuple((-v[1],-v[0]) for v in a)
def sub(a,b):return add(a,neg(b))
def product(a,b):
    out=[(F(0),F(0)) for _ in range(len(a)+len(b)-1)]
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            values=(x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1]);low,high=out[i+j]
            out[i+j]=(low+min(values),high+max(values))
    return tuple(out)
def vsub(a,b):return tuple(sub(x,y) for x,y in zip(a,b))
def cross(a,b):return tuple(sub(product(a[i],b[j]),product(a[j],b[i])) for i,j in ((1,2),(2,0),(0,1)))
def dot_axis(a,axis):
    out=[(F(0),F(0)) for _ in range(len(a[0]))]
    for p,w in zip(a,axis):
        for i,(lo,hi) in enumerate(p):
            lo,hi=(lo*w,hi*w) if w>=0 else (hi*w,lo*w)
            out[i]=(out[i][0]+lo,out[i][1]+hi)
    return tuple(out)
def positive(p):
    degree=len(p)-1
    return all(sum(p[i][0]*F(comb(k,i),comb(degree,i)) for i in range(k+1))>0 for k in range(degree+1))
def exact_cross(a,b):return tuple(a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1)))
def magnitude(p):return sum(max(abs(lo),abs(hi)) for lo,hi in p)
def computed_normal_positive(points,axis,index):
    # Enclose the actual floating subtraction/product/subtraction sequence
    # producing the nonzero raw-normal component. The other two components
    # are exactly zero when the fixed perpendicular coordinates are equal.
    u=F(1,2**53);gamma=3*u/(1-3*u)
    a=vsub(points[1],points[0]);b=vsub(points[2],points[0])
    def widen(edge,left,right):
        out=[]
        for e,p,q in zip(edge,left,right):
            error=u*(magnitude(p)+magnitude(q))/(1-u)+F(1,2**1074)
            out.append(((e[0][0]-error,e[0][1]+error),*e[1:]))
        return tuple(out)
    a=widen(a,points[1],points[0]);b=widen(b,points[2],points[0])
    raw=cross(a,b)[index];j,k=((1,2),(2,0),(0,1))[index]
    error=gamma*(magnitude(a[j])*magnitude(b[k])+magnitude(a[k])*magnitude(b[j]))+F(3,2**1074)
    raw=((raw[0][0]-error,raw[0][1]+error),*raw[1:])
    # A wide machine-derived representability envelope protects the actual
    # square/sum/sqrt normalization, separately from geometric tolerances.
    minimum=F(float(np.finfo(float).tiny**.25))
    maximum=F(float(np.finfo(float).max**.25))
    signed_raw=raw if axis[index]>0 else neg(raw)
    if not positive(sub(signed_raw,((minimum,minimum),*( (F(0),F(0)) for _ in raw[1:])))):return False
    if magnitude(raw)>maximum:return False
    signed=dot_axis((raw,), (axis[index],))
    return positive(signed)


def map_rows(reference, A, first, last):
    """Exact fixed-map polynomial + uniform 3-op q and sparse-dot enclosure.

    Binary reference/map are authoritative. gamma bounds include q(t) formed
    by subtraction/multiply/add, sparse row products/sums and reference add.
    Zero products plus reference are exactly reference, avoiding fictitious
    uncertainty in the explicitly fixed target-plane coordinate.
    """
    u=F(1,2**53);gamma3=3*u/(1-3*u);result=[];exact=[]
    A=A.tocsr()
    q0=tuple(F(float(x)) for x in first);q1=tuple(F(float(x)) for x in last)
    for k,reference in enumerate(np.asarray(reference).reshape(-1)):
        lo,hi=A.indptr[k:k+2];coeff=tuple((int(j),F(float(v))) for j,v in zip(A.indices[lo:hi],A.data[lo:hi]))
        a=F(float(reference))+sum(v*q0[j] for j,v in coeff);b=sum(v*(q1[j]-q0[j]) for j,v in coeff)
        magnitude=sum(abs(v)*max(abs(q0[j]),abs(q1[j])) for j,v in coeff)
        input_error=sum(abs(v)*(gamma3*2*(abs(q0[j])+abs(q1[j]))+F(3,2**1074)) for j,v in coeff if q0[j] or q1[j])
        if magnitude==0 and input_error==0:error=F(0)
        else:
            n=2*len(coeff)+2;gamma=n*u/(1-n*u)
            error=gamma*(abs(F(float(reference)))+magnitude)+input_error*(1+gamma)+F(n,2**1074)
        result.append(((a-error,a+error),(b,b)));exact.append((a,b))
    return tuple(tuple(result[3*i+j] for j in range(3)) for i in range(len(result)//3)),tuple(tuple(exact[3*i+j] for j in range(3)) for i in range(len(exact)//3))


def mapped_path(adapter,first,last):
    return map_rows(adapter.reference_positions,adapter.work_map,first,last)


def material_point_paths(owner,first,last):
    from scipy import sparse
    # These stored binary maps/reference points are the actual dense-dot
    # operands used by friction, independently of approximate barycentric
    # partition of unity. Count ALL dense columns, including zero products.
    result=[]
    for reference,mapping in zip(owner.point_reference,owner.point_maps):
        dense=sparse.csr_matrix(mapping)
        paths,_=map_rows(reference,dense,first[owner.work_columns],last[owner.work_columns])
        # map_rows counts nonzeros. Dense BLAS accumulates at most n products:
        # widen by the difference in the conservative operation count.
        u=F(1,2**53);n=2*mapping.shape[1]+2;gamma=n*u/(1-n*u)
        widened=[]
        for k,p in enumerate(paths[0]):
            magnitude=sum(abs(F(float(v)))*max(abs(F(float(a))),abs(F(float(b))))
                          for v,a,b in zip(mapping[k],first[owner.work_columns],last[owner.work_columns]))
            error=gamma*(abs(F(float(reference[k])))+magnitude)+F(n,2**1074)
            widened.append(((p[0][0]-error,p[0][1]+error),*p[1:]))
        result.append(tuple(widened))
    return result


def full_coverage(owner,first,last):
    first=np.asarray(first,float);last=np.asarray(last,float)
    if first.shape!=(owner.n_dof,) or last.shape!=first.shape or not np.isfinite(first).all() or not np.isfinite(last).all():return False
    # All map operands/intermediate magnitudes are kept well below overflow.
    cap=np.finfo(float).max**.125
    arrays=[first,last]
    for adapter in (*owner.source_adapters,*owner.target_adapters.values()):
        arrays.extend((adapter.reference_positions,adapter.work_map.data))
    arrays.extend(owner.point_reference);arrays.extend(owner.point_maps)
    if any(not np.isfinite(a).all() or np.max(np.abs(a),initial=0.)>cap for a in arrays):return False
    target={e:mapped_path(a,first,last) for e,a in owner.target_adapters.items()}
    reference=next(iter(owner.target_adapters.values())).reference_positions
    origin=tuple(F(float(x)) for x in reference[0]);edges=[tuple(F(float(x))-o for x,o in zip(p,origin)) for p in reference[1:]]
    axis=exact_cross(*edges)
    nonzero=tuple(i for i,v in enumerate(axis) if v)
    if len(nonzero)!=1:return False
    perpendicular=nonzero[0]
    # Exact fixed-map target field must keep the declared plane normal. This
    # finite predicate deliberately refuses even safe varying-plane paths.
    anchor=next(iter(target.values()))[1][0]
    enclosed_anchor=next(iter(target.values()))[0][0]
    vertices={};boundary=[]
    for e,a in owner.target_adapters.items():
        points,exact=target[e]
        for node,p,nominal in zip(a.node_ids,points,exact):
            # Only a FIXED exact axis-aligned target coordinate is promoted.
            # Oblique or normally translated maps need a different normal/
            # frame roundoff enclosure and are conservatively refused here.
            if nominal[perpendicular][1]!=0:return False
            for degree in range(2):
                if sum((nominal[j][degree]-anchor[j][degree])*axis[j] for j in range(3))!=0:return False
            # Roundoff must also preserve the declared plane, rather than
            # asserting constancy only for ideal polynomial coordinates. This
            # strict sufficient predicate is conservative/axis-anisotropic;
            # unresolved tilted/translated maps are not admitted silently.
            if any(lo!=0 or hi!=0 for lo,hi in dot_axis(vsub(p,enclosed_anchor),axis)):return False
            if node in vertices and vertices[node]!=p:return False
            vertices[node]=p
        if not computed_normal_positive(points,axis,perpendicular):return False
        for i,node in enumerate(a.node_ids):
            other=a.node_ids[(i+1)%3]
            if tuple(sorted((node,other))) in owner.boundary_edges:boundary.append((node,other))
    source=[p for a in owner.source_adapters for p in mapped_path(a,first,last)[0]]
    source.extend(material_point_paths(owner,first,last))
    for a,b in boundary:
        origin=vertices[a];edge=vsub(vertices[b],origin)
        for node,p in vertices.items():
            if node in (a,b):continue
            if not positive(dot_axis(cross(edge,vsub(p,origin)),axis)):return False
        for p in source:
            if not positive(dot_axis(cross(edge,vsub(p,origin)),axis)):return False
    return True
