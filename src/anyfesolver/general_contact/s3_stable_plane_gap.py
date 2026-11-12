"""Stable primal preparation of the unchanged affine P1 plane-gap expression.

This is opt-in arithmetic for finite fixed-planar rigid contact. Fraction
determinants retain the exact binary64 input differences/products in the small
corner-plane numerators; Jet2 supplies both-body derivatives of the same
expression. No geometric normal, force, support or physical coordinate is
adjusted. Ordinary overlap/chart admission remains the caller's responsibility.
"""
from fractions import Fraction
from anyfesolver._ge_beam3_mixed_ad import Jet2

def _sub(a,b):return [x-y for x,y in zip(a,b)]
def _dot(a,b):return sum((x*y for x,y in zip(a,b)))
def _cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]

def stable_corner_gap_jets(source_jets,target_jets,source,target):
    """Return affine corner heights with full source/target Jet2 derivatives.

    The compensated numerator, squared normal norm and mutual denominator are
    rounded once from exact binary64-input rational expressions. Their existing
    analytic derivatives remain Jet2 arrays; subsequent division/product uses
    these stable primals consistently for gradient and Hessian coefficients.
    """
    from anyfesolver._ge_beam3_mixed_ad import sqrt as jet_sqrt
    s=[[Fraction(float(v)) for v in row] for row in source]
    t=[[Fraction(float(v)) for v in row] for row in target]
    ns=_cross(_sub(s[1],s[0]),_sub(s[2],s[0]))
    nt=_cross(_sub(t[1],t[0]),_sub(t[2],t[0]))
    square=_dot(ns,ns);denominator=_dot(nt,ns)
    if square<=0 or denominator<=0:
        raise ValueError('stable plane gap requires a positive source/mutual chart')
    js=_cross(_sub(source_jets[1],source_jets[0]),_sub(source_jets[2],source_jets[0]))
    jt=_cross(_sub(target_jets[1],target_jets[0]),_sub(target_jets[2],target_jets[0]))
    square_jet=_dot(js,js);denominator_jet=_dot(jt,js)
    square_jet=Jet2(float(square),square_jet.gradient,square_jet.hessian)
    denominator_jet=Jet2(float(denominator),denominator_jet.gradient,denominator_jet.hessian)
    length=jet_sqrt(square_jet)
    result=[]
    for point,exact in zip(source_jets,s):
        numerator=_dot(jt,_sub(point,target_jets[0]))
        numerator=Jet2(float(_dot(nt,_sub(exact,t[0]))),numerator.gradient,numerator.hessian)
        result.append(numerator*length/denominator_jet)
    return result
