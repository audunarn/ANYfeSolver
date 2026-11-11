"""Outward-dyadic sign filter for an exactly represented shifted pencil.

Every endpoint is an integer in units of 2**-precision. No floating-point
arithmetic determines a pivot sign. An unresolved enclosure returns None;
the caller must use exact arithmetic or reject, never guess a sign.
This is not a continuum or pre-rounding operator certificate.
"""
from time import monotonic
import numpy as np


def _multiply(a,b,scale):
    products=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
    return min(products)//scale,-((-max(products))//scale)


def _divide(a,b,scale):
    if b[0]<=0<=b[1]: raise ZeroDivisionError('interval pivot contains zero')
    fractions=((a[0]*scale,b[0]),(a[0]*scale,b[1]),(a[1]*scale,b[0]),(a[1]*scale,b[1]))
    return min(n//d for n,d in fractions),max(-((-n)//d) for n,d in fractions)


def interval_inertia(h,mass,shift,check=lambda:None,*,precision=256,dimension_limit=64):
    started=monotonic()
    def guard():
        check()
        if monotonic()-started>30.: raise TimeoutError('dyadic shifted inertia deadline')
    guard()
    if type(precision) is not int or precision not in (256,512): raise ValueError('admitted dyadic precision required')
    if type(dimension_limit) is not int or dimension_limit not in (64,96,160): raise ValueError('explicit admitted exact-inertia dimension limit required')
    if (h.ndim!=2 or h.shape[0]!=h.shape[1] or not 1<=len(h)<=dimension_limit or mass.shape!=h.shape
            or not np.isfinite(h).all() or not np.isfinite(mass).all() or not np.isfinite(shift)
            or not np.array_equal(h,h.T) or not np.array_equal(mass,mass.T)):
        raise ValueError('bounded finite symmetric shifted pencil required')
    scale=1<<precision; sn,sd=float(shift).as_integer_ratio(); a=[]
    for row,other in zip(h,mass):
        guard(); entries=[]
        for x,y in zip(row,other):
            xn,xd=float(x).as_integer_ratio(); yn,yd=float(y).as_integer_ratio()
            d=max(xd,sd*yd); n=(xn*(d//xd)-sn*yn*(d//(sd*yd)))*scale
            entries.append((n//d,-((-n)//d)))
        a.append(entries)
    positive=negative=0
    while a:
        guard(); n=len(a)
        candidates=[i for i in range(n) if a[i][i][0]>0 or a[i][i][1]<0]
        if not candidates:
            if all(entry==(0,0) for row in a for entry in row): return positive,negative,n
            return None
        pivot=max(candidates,key=lambda i:min(abs(a[i][i][0]),abs(a[i][i][1])))
        order=[pivot]+[i for i in range(n) if i!=pivot]
        a=[[a[i][j] for j in order] for i in order]; d=a[0][0]
        positive+=int(d[0]>0); negative+=int(d[1]<0)
        quotients=[_divide(a[i][0],d,scale) for i in range(1,n)]
        reduced=[[(0,0)]*(n-1) for _ in range(n-1)]
        for i in range(1,n):
            guard()
            for j in range(i,n):
                lo,hi=_multiply(quotients[i-1],a[0][j],scale)
                entry=(a[i][j][0]-hi,a[i][j][1]-lo)
                reduced[i-1][j-1]=reduced[j-1][i-1]=entry
        a=reduced
    guard()
    return positive,negative,0
