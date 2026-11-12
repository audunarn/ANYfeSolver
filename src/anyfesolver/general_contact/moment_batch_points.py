"""Same physical moment integrand evaluated in25 independent point lanes."""
from __future__ import annotations
import numpy as np
from anyfesolver.e4_pl_enriched import bubble_basis
from .moment_batch_ad import PointJet,sqrt,choose
from .enriched_overlap import EnrichedGeometryError,SIZE

POINTS=25

def make_integrand(ref,q,t,bb,body_normal,normal,tangent,second,triangle,area2,scale,resolution):
    def cast(x):return PointJet.broadcast(x,POINTS)if hasattr(x,'gradient')else PointJet.constant(x,POINTS,SIZE)
    def vector(v):return [cast(x)for x in v]
    q=[vector(p)for p in q];t=[vector(p)for p in t];bb=vector(bb)
    body_normal=vector(body_normal);normal=vector(normal);tangent=vector(tangent);second=vector(second)
    triangle=[vector(p)for p in triangle];area2=cast(area2)
    ref=[vector(p)for p in ref]
    def const(x):return PointJet.constant(x,POINTS,SIZE)
    def dot(a,b):return sum((x*y for x,y in zip(a,b)),const(0))
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
    def unit(a):
        length=sqrt(dot(a,a))
        return [x/length for x in a]
    def interpolate(n,points):return [sum((n[i]*points[i][k]for i in range(len(n))),const(0))for k in range(len(points[0]))]
    def shape(r,s):
        return (((1-r)*(1-s)/4,(1+r)*(1-s)/4,(1+r)*(1+s)/4,(1-r)*(1+s)/4),
                (-(1-s)/4,(1-s)/4,(1+s)/4,-(1+s)/4),
                (-(1-r)/4,-(1+r)/4,(1+r)/4,(1-r)/4))
    def physical(r,s):
        n,nr,ns=shape(r,s);b,br,bs=bubble_basis(r,s);out=[]
        for basis,extra in ((n,b),(nr,br),(ns,bs)):
            nodal=interpolate(basis,q);amount=sum((extra[i]*bb[i]for i in range(4)),const(0))
            out.append([nodal[k]+amount*body_normal[k]for k in range(3)])
        return n,out[0],out[1],out[2]
    def project(p):return [dot(sub(p,q[0]),tangent),dot(sub(p,q[0]),second)]
    def maxima(v):
        return (np.maximum.reduce([np.abs(x.value)for x in v]),
                np.maximum.reduce([np.max(np.abs(x.gradient),axis=1)for x in v]),
                np.maximum.reduce([np.max(np.abs(x.hessian),axis=(1,2))for x in v]))

    def integrand(xys):
        if len(xys)!=POINTS:raise EnrichedGeometryError('exact25point Duffy lane count required')
        xy=[PointJet.stack([p[k]for p in xys])for k in range(2)]
        r,s=const(0),const(0);active=np.ones(POINTS,dtype=bool)
        # Each lane retains the ORIGINAL update-before-break semantics.
        # Completed lanes freeze while later lanes continue their own updates.
        for iteration in range(12):
            n,p,pr,ps=physical(r,s);residual=sub(project(p),xy)
            ar=[dot(pr,tangent),dot(pr,second)];ass=[dot(ps,tangent),dot(ps,second)]
            det=cross2(ar,ass)
            if np.any(det.value[active]<=resolution):raise EnrichedGeometryError('folded enriched projected skin')
            next_r,next_s=r-cross2(residual,ass)/det,s-cross2(ar,residual)/det
            r,s=choose(active,next_r,r),choose(active,next_s,s)
            if iteration>=2:
                value,gradient,hessian=maxima(residual)
                done=(value<=1e-12*scale)&(gradient<=1e-10)&(hessian<=1e-8/scale)
                active &= ~done
                if not np.any(active):break
        n,p,pr,ps=physical(r,s);residual=sub(project(p),xy)
        value,gradient,hessian=maxima(residual)
        if (np.any(value>1e-11*scale)or np.any(gradient>1e-9)or np.any(hessian>1e-7/scale)
                or np.any(np.maximum(np.abs(r.value),np.abs(s.value))>1+1e-10)):
            raise EnrichedGeometryError('enriched projected inversion unresolved')
        cross_skin=cross(pr,ps);projected_jac=dot(cross_skin,normal)
        if np.any(projected_jac.value<=resolution):raise EnrichedGeometryError('nonpositive physical skin Jacobian')
        skin_normal=unit(cross_skin);v=sub(xy,triangle[0])
        l1=cross2(v,sub(triangle[2],triangle[0]))/area2
        l2=cross2(sub(triangle[1],triangle[0]),v)/area2
        target_point=interpolate((1-l1-l2,l1,l2),t)
        gap=dot(skin_normal,sub(p,target_point));_,nr,ns=shape(r,s)
        reference_cross=cross(interpolate(nr,ref),interpolate(ns,ref))
        reference_jac=sqrt(dot(reference_cross,reference_cross));weight=reference_jac/projected_jac
        return [weight*x for x in n]+[weight*x*gap for x in n]+[weight*x*y for x in n for y in n]
    return integrand
