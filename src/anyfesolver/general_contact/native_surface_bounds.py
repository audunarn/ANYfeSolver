"""Conservative current boxes for native round material patches.

The bound contains the whole declared arc and both half-cell traces, independently
of quadrature. A circumscribed full round section is used for search only: this
does not replace the material surface, certify orientation, prove gap or sweep.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import cos, pi, sqrt

import numpy as np

from anyfesolver._ge_beam3_deterministic_reduction import deterministic_norm
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from .native_retained_station import _readonly
from .native_surface_quadrature import NativeRoundSurfaceRule


def _restrict_q2(lanes, left, right):
    """Quadratic blossom at exact rational material parameters."""
    l,r=left,right
    return tuple(((1-l)**2*a+2*l*(1-l)*b+l*l*c,
                  (1-l)*(1-r)*a+(l*(1-r)+r*(1-l))*b+l*r*c,
                  (1-r)**2*a+2*r*(1-r)*b+r*r*c) for a,b,c in lanes)


def _rotation_gain(frame):
    """Upper spectral norm from the Gram residual, including its arithmetic."""
    eps = np.finfo(float).eps
    residual = float(deterministic_norm(frame.T@frame-np.eye(3)))
    residual += 128*eps*(1+deterministic_norm(frame)**2)
    gain = sqrt(1+residual)*(1+128*eps)
    if not np.isfinite(gain):
        raise ValueError('native material frame bound overflow')
    return gain


def _reference_radial_gain(reference):
    """Include admitted nodal nonorthogonality in shortest transport.

    For Q=I+K+K²/(1+c), on the cross plane ||Q||²=1+
    s²*(||a||²||b||²-1)/(1+c)² for unclipped c. The conservative
    factor4 also covers clipping at c=1. The source rejects denominators
    below 1+cos(0.9*pi). Roll uses the normalized reference tangent.
    """
    eps=np.finfo(float).eps
    margin=1+cos(.9*pi)
    gains=[]
    for frame in reference.nodal_triads:
        e=float(np.linalg.norm(frame.T@frame-np.eye(3)))
        e+=128*eps*(1+float(np.linalg.norm(frame))**2)
        transport=sqrt(1+4*(1+e+128*eps)*(e+128*eps)/margin**2)
        gains.append(_rotation_gain(frame)*transport*(1+1024*eps/margin**2))
    gain=max(gains)
    if not np.isfinite(gain):
        raise ValueError('native material reference frame bound overflow')
    return gain


@dataclass(frozen=True)
class NativeRoundPatchBounds:
    lower: np.ndarray
    upper: np.ndarray
    cells: tuple[int, ...]
    roundoff_padding_m: float
    profile: str = 'NATIVE_Q2_RETAINED_CURRENT_BERNSTEIN_ROUND_ENVELOPE_DEV_V1'

    def as_search_box(self):
        return _readonly(np.r_[self.lower, self.upper])


def native_round_patch_bounds(rule, positions, position_low, cell_rotations):
    """Reference-owned Q2 lift + current chord + circumscribed section.

    For each cell, x_c(t)=(1-t)p_L+t p_R+Uc*b*t*(t-1)/2.
    Its Bernstein control is (p_L,(p_L+p_R)/2-Uc*b/4,p_R).
    Restriction is exact rational arithmetic on the supplied binary coordinates;
    b=X0-2X1+X2 comes from that same centered reference. Bounds use the hull,
    not quadrature or endpoints alone. The material radial part has norm at most
    r*||Uc|| times the reference-frame gain, including admitted nodal Gram
    residuals amplified by shortest transport; roll arithmetic is also bounded.

    Padding covers bounded floating evaluation of the admitted frame transport
    (half-cell turn<0.9*pi), lift and position. Overflow rejects. Not an exact
    collision certificate or dynamic path enclosure.
    """
    if type(rule) is not NativeRoundSurfaceRule:
        raise TypeError('exact native round surface rule required')
    high = _array(positions, (3, 3), 'native bounds nodal positions')
    low = _array(position_low, (3, 3), 'native bounds nodal low parts')
    rotations = _frames(cell_rotations, 2, 'native bounds cell rotations')
    # Bind the same definitions, normalized coordinate pairs and owned geometry.
    # Search bounds never infer validity from finite sample containment.
    rule.evaluate(high, low, rotations)
    reference = rule.samples[0].station.reference
    X = [[Fraction(float(v)) for v in row] for row in reference.coordinates]
    b = [X[0][j]-2*X[1][j]+X[2][j] for j in range(3)]
    p = [[Fraction(float(h))+Fraction(float(l)) for h,l in zip(hr,lr)]
         for hr,lr in zip(high,low)]
    radius = float(rule.definition['radius'])
    xi_lo, xi_hi = rule.definition['xi_range']
    controls = []
    cells = []
    # Frobenius residual bounds the spectral norm without an SVD rounding claim.
    eps = np.finfo(float).eps
    radial_gain = _reference_radial_gain(reference)
    scale = max(1., radius, float(np.max(np.abs(high))),
                float(np.max(np.abs(low))), float(np.max(np.abs(reference.coordinates))))
    padding = 4096*eps*scale
    for cell, left, right in ((0,max(xi_lo,-1.),min(xi_hi,0.)),
                             (1,max(xi_lo,0.),min(xi_hi,1.))):
        if right <= left:
            continue
        cells.append(cell)
        U = [[Fraction(float(v)) for v in row] for row in rotations[cell]]
        turn = [sum(U[i][j]*b[j] for j in range(3)) for i in range(3)]
        lanes = [(p[cell][i],(p[cell][i]+p[cell+1][i])/2-turn[i]/4,
                  p[cell+1][i]) for i in range(3)]
        l=Fraction(float(left))+(1 if cell==0 else 0)
        r=Fraction(float(right))+(1 if cell==0 else 0)
        restricted = _restrict_q2(lanes, l, r)
        gain = _rotation_gain(rotations[cell])
        extent = Fraction(radius*radial_gain*gain)+Fraction(padding)
        controls.append(([min(row)-extent for row in restricted],
                         [max(row)+extent for row in restricted]))
    try:
        lower = np.array([np.nextafter(float(min(c[0][i] for c in controls)), -np.inf)
                          for i in range(3)])
        upper = np.array([np.nextafter(float(max(c[1][i] for c in controls)), np.inf)
                          for i in range(3)])
    except (OverflowError, ValueError) as exc:
        raise ValueError('native material search bound overflow') from exc
    if not np.all(np.isfinite(np.r_[lower,upper])):
        raise ValueError('native material search bound overflow')
    return NativeRoundPatchBounds(_readonly(lower),_readonly(upper),tuple(cells),padding)


def native_round_patch_swept_bounds(rule, positions, position_low, cell_rotations,
                                    translations, angular):
    """Whole patch on the existing affine/Exp path, for search only.

    Translation is affine in alpha; half-cell lift and radial section rotate.
    A circumsphere of their summed reference radii encloses every spatial
    exponential rotation, not just endpoints. Angular increments obey the
    same <0.9*pi limit as material_path_pose. No collision-time, orientation,
    normal-side or physical-gap certificate is supplied.
    """
    current = native_round_patch_bounds(rule,positions,position_low,cell_rotations)
    move = _array(translations,(3,3),'native bounds translations')
    spin = _array(angular,(2,3),'native bounds angular increments')
    if np.max(np.linalg.norm(spin,axis=1)) >= .9*pi:
        raise ValueError('native material path requires subdivision: angular increment')
    high = np.asarray(positions); low = np.asarray(position_low)
    reference = rule.samples[0].station.reference
    X = [[Fraction(float(v)) for v in row] for row in reference.coordinates]
    b = [X[0][j]-2*X[1][j]+X[2][j] for j in range(3)]
    p = [[Fraction(float(h))+Fraction(float(l)) for h,l in zip(hr,lr)]
         for hr,lr in zip(high,low)]
    delta = [[Fraction(float(v)) for v in row] for row in move]
    from .capped_cylinder import _sqrt_bound
    bnorm = _sqrt_bound(sum(v*v for v in b),upper=True)
    eps = np.finfo(float).eps
    radial = Fraction(float(rule.definition['radius'])*_reference_radial_gain(reference))
    # Same current rotation residual; Exp path evaluation adds floating padding.
    gain = max(_rotation_gain(f) for f in np.asarray(cell_rotations))*(1+128*eps)
    end_scale = max(1.,float(np.max(np.abs(move))),float(np.max(np.abs(high))),
                    float(np.max(np.abs(low))),float(np.max(np.abs(reference.coordinates))))
    padding = max(current.roundoff_padding_m,8192*eps*end_scale)
    xi_lo,xi_hi=rule.definition['xi_range']
    hulls=[]
    for cell,left,right in ((0,max(xi_lo,-1.),min(xi_hi,0.)),
                            (1,max(xi_lo,0.),min(xi_hi,1.))):
        if right<=left:continue
        l = Fraction(float(left))+(1 if cell==0 else 0)
        r = Fraction(float(right))+(1 if cell==0 else 0)
        extrema = [l*(1-l),r*(1-r)]
        if l<=Fraction(1,2)<=r:extrema.append(Fraction(1,4))
        lift = max(extrema)*bnorm/2
        extent = Fraction(gain)*(lift+radial)+Fraction(padding)
        limits=[[],[]]
        for i in range(3):
            vals = [(1-t)*(p[cell][i]+a*delta[cell][i])+
                    t*(p[cell+1][i]+a*delta[cell+1][i])
                    for a in (0,1) for t in (l,r)]
            limits[0].append(min(vals)-extent)
            limits[1].append(max(vals)+extent)
        hulls.append(limits)
    try:
        lower=np.array([np.nextafter(float(min(h[0][i] for h in hulls)),-np.inf)
                        for i in range(3)])
        upper=np.array([np.nextafter(float(max(h[1][i] for h in hulls)),np.inf)
                        for i in range(3)])
    except (OverflowError,ValueError) as exc:
        raise ValueError('native material swept bound overflow') from exc
    if not np.all(np.isfinite(np.r_[lower,upper])):
        raise ValueError('native material swept bound overflow')
    return NativeRoundPatchBounds(_readonly(lower),_readonly(upper),current.cells,padding,
        'NATIVE_Q2_RETAINED_AFFINE_EXP_SPHERE_SWEEP_DEV_V1')
