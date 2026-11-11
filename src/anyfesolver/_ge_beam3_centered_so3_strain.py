"""Explicit centered retained SO3 arithmetic successor; no legacy P5 edits.

Effective SO3 base and its complete Jet2 increments define the same work value,
first and second variations. Literal rounded-matrix bit equivalence is not claimed.
"""
import math
import numpy as np
from ._ge_beam3_mixed_ad import (Jet2, _exp_coefficients, constant_matrix,
    identity, matmul, skew, transpose)
from ._ge_beam3_p5.compensated import sum_jets
from ._ge_beam3_p5.compensated_coordinates import split_sum

POLICY='GE_BEAM3_CENTERED_STABLE_SO3_AND_PRECISE_WORK_DEV_V1'
ROUND_OFF_ULPS=64


def base_delta(rotation):
    """Stable effective U-I, with a machine-roundoff SO3 certificate near I.

    For exact SO3, S=(U-U.T)/2,c=(trU-1)/2 gives U-I=S+S^2/(1+c).
    c>.5 has denominator>1.5. Outside that branch use the historical difference.
    The certificate includes an8eps arithmetic allowance for measured3x3 products.
    No projection/normalization of an arbitrarily distorted matrix is admitted.
    """
    u=np.asarray(rotation,dtype=np.float64)
    if u.shape!=(3,3) or not np.isfinite(u).all():raise ValueError('finite3x3 SO3 base required')
    cosine=(math.fsum(float(u[i,i]) for i in range(3))-1.)/2
    if cosine>.5:
        eps=np.finfo(float).eps
        if (np.max(np.abs(u))>1.+ROUND_OFF_ULPS*eps or
            np.linalg.norm(u.T@u-np.eye(3),ord=np.inf)+8*eps>ROUND_OFF_ULPS*eps or
            abs(float(np.linalg.det(u))-1.)+8*eps>ROUND_OFF_ULPS*eps):
            raise ValueError('stable SO3 base exceeds machine-roundoff certificate')
        s=(u-u.T)/2
        result=s+s@s/(1.+cosine)
    else:result=u-np.eye(3)
    if not np.isfinite(result).all():raise ValueError('finite stable SO3 difference required')
    return result


def rotation_jets(rotation,vector):
    """Return effective madeU, made(U-I), captured base(U-I) without subtraction."""
    size=vector[0].gradient.size;base=base_delta(rotation)
    d=constant_matrix(base,size)
    argument=sum((v*v for v in vector),Jet2.constant(0.,size))
    a,b=_exp_coefficients(argument);cross=skew(vector);square=matmul(cross,cross)
    e=[[a*cross[i][j]+b*square[i][j] for j in range(3)] for i in range(3)]
    product=matmul(e,d)
    made_d=[[d[i][j]+e[i][j]+product[i][j] for j in range(3)] for i in range(3)]
    unit=identity(size)
    made=[[unit[i][j]+made_d[i][j] for j in range(3)] for i in range(3)]
    return made,made_d,base


def strain(made_delta,reference,position_high,position_low,variables,left,right):
    """Complete Jet2 successor of compensated_strain's identical chord law.

    z=change+D.T*reference_chord+D.T*change, retaining both coordinate parts.
    No value-only adjustment: derivatives of D/coordinates remain in every term.
    """
    size=variables[0].gradient.size;base=[];change=[]
    for i in range(3):
        baseline=(float(reference[right,i]),-float(reference[left,i]))
        base.append(split_sum(baseline))
        terms=(float(position_high[right,i]),float(position_low[right,i]),
            -float(position_high[left,i]),-float(position_low[left,i]),
            variables[6*right+i].value,-variables[6*left+i].value,-baseline[0],-baseline[1])
        high,low=split_sum(terms);derivative=variables[6*right+i]-variables[6*left+i]
        change.append((Jet2(high,derivative.gradient,derivative.hessian),Jet2.constant(low,size)))
    dt=transpose(made_delta);result=[]
    for i in range(3):
        terms=list(change[i])
        for k in range(3):
            terms.extend(dt[i][k]*v for v in base[k]);terms.extend(dt[i][k]*v for v in change[k])
        result.append(sum_jets(terms))
    return result
