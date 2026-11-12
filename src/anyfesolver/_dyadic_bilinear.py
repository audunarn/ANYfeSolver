"""Exact supplied-binary64 bilinear reassembly, rounded once per output entry.

Each finite binary64 matrix is represented as integer numerators over one
power-of-two denominator. Python integers retain the complete products and
sums, including cancellation between material and geometric work. This is
not interval arithmetic or a claim that the supplied beam data are exact.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class _Matrix:
    rows: tuple
    power: int


def _capture(a,checkpoint):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all(): raise ValueError('finite nonempty dyadic matrix')
    pairs=[]; power=0
    for row in a:
        checkpoint(); made=[]
        for value in row:
            n,d=float(value).as_integer_ratio(); exponent=d.bit_length()-1
            power=max(power,exponent); made.append((n,exponent))
        pairs.append(made)
    return _Matrix(tuple(tuple(n<<(power-e) for n,e in row) for row in pairs),power)


def _transpose(a): return _Matrix(tuple(zip(*a.rows)),a.power)


def _multiply(a,b,checkpoint):
    if len(a.rows[0])!=len(b.rows): raise ValueError('matching dyadic dimensions')
    columns=tuple(zip(*b.rows)); rows=[]
    for row in a.rows:
        checkpoint(); rows.append(tuple(sum(x*y for x,y in zip(row,col)) for col in columns))
    return _Matrix(tuple(rows),a.power+b.power)


def _add(a,b):
    if len(a.rows)!=len(b.rows) or len(a.rows[0])!=len(b.rows[0]): raise ValueError('matching dyadic sum shapes')
    power=max(a.power,b.power); left=power-a.power; right=power-b.power
    return _Matrix(tuple(tuple((x<<left)+(y<<right) for x,y in zip(row,col))
        for row,col in zip(a.rows,b.rows)),power)


def _rounded(a):
    denominator=1<<a.power; rows=[]
    for row in a.rows:
        made=[]
        for value in row:
            try: result=value/denominator
            except OverflowError as exc: raise ValueError('unrepresentable dyadic result') from exc
            if not np.isfinite(result) or (result==0. and value!=0):
                raise ValueError('unrepresentable dyadic result')
            made.append(result)
        rows.append(made)
    return np.array(rows)


def reassemble_exact_binary64(f,g,b,mapping,checkpoint=lambda:None):
    """Complete V.T*(F.T*F+G)*V and V.T*B.T*B*V, not a low-rank fit."""
    if (np.ndim(mapping)!=2 or not 1<=mapping.shape[0]<=256 or not 1<=mapping.shape[1]<=256
            or np.ndim(f)!=2 or np.ndim(b)!=2 or not 1<=len(f)<=8192 or not 1<=len(b)<=8192
            or f.shape[1]!=mapping.shape[0] or b.shape[1]!=mapping.shape[0]
            or np.shape(g)!=(mapping.shape[0],)*2):
        raise ValueError('bounded matching bilinear dimensions')
    ff,gg,bb,v=(_capture(a,checkpoint) for a in (f,g,b,mapping))
    strain=_multiply(ff,v,checkpoint); speed=_multiply(bb,v,checkpoint)
    material=_multiply(_transpose(strain),strain,checkpoint)
    geometric=_multiply(_transpose(v),_multiply(gg,v,checkpoint),checkpoint)
    mass=_multiply(_transpose(speed),speed,checkpoint)
    checkpoint()
    return _rounded(_add(material,geometric)),_rounded(mass)
