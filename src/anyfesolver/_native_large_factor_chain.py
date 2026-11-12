"""Sparse exact integer products; reassociate before dense modal congruence.

No binary64 Gram matrix is formed. Integer products/sums are exact, and each
final projected entry is rounded once, as in the historical implementation.
"""
from math import fsum
import numpy as np
from ._dyadic_bilinear import _Matrix,_capture,_transpose,_add,_rounded


def multiply(a,b,checkpoint):
    if len(a.rows[0])!=len(b.rows):raise ValueError('matching exact factor dimensions')
    left=[tuple((k,x) for k,x in enumerate(row) if x) for row in a.rows]
    columns=tuple(zip(*b.rows))
    right=[tuple((k,x) for k,x in enumerate(col) if x) for col in columns]
    rows=[]
    for i,row in enumerate(left):
        checkpoint();values=[]
        for j,col in enumerate(right):
            if len(row)<=len(col):value=sum(x*columns[j][k] for k,x in row)
            else:value=sum(a.rows[i][k]*x for k,x in col)
            values.append(value)
        rows.append(tuple(values))
    return _Matrix(tuple(rows),a.power+b.power)


def gram(a,checkpoint):
    width=len(a.rows[0]);rows=[[0]*width for _ in range(width)]
    for row in a.rows:
        checkpoint();items=[(i,x) for i,x in enumerate(row) if x]
        for k,(i,x) in enumerate(items):
            for j,y in items[k:]:rows[i][j]+=x*y
    for i in range(width):
        for j in range(i):rows[i][j]=rows[j][i]
    return _Matrix(tuple(map(tuple,rows)),2*a.power)


def reassemble(left,right,g,b,mapping,checkpoint):
    ll,rr,gg,bb,v=(_capture(a,checkpoint) for a in (left,right,g,b,mapping))
    material=gram(multiply(ll,rr,checkpoint),checkpoint)
    signed=_add(material,gg);mass=gram(bb,checkpoint)
    vt=_transpose(v)
    h=multiply(vt,multiply(signed,v,checkpoint),checkpoint)
    m=multiply(vt,multiply(mass,v,checkpoint),checkpoint)
    checkpoint();return _rounded(h),_rounded(m)


def sparse_float_product(a,b,dot_terms,checkpoint):
    rows=[];columns=b.T
    for row in a:
        checkpoint();indices=np.flatnonzero(row);values=[]
        left=row[indices]
        for column in columns:values.append(fsum(dot_terms(left,column[indices])))
        rows.append(values)
    return np.array(rows)
