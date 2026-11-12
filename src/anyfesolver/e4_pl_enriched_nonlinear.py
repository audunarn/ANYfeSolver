"""Separate elastic nonlinear physical B4 enrichment, development only.

The exact current nodal law belongs to the existing assembly caller. This
module supplies ONLY the stationary-potential difference Psi(q,b)-Psi(q,0).
All 28 displacement coordinates remain; the same43 internal fields are
eliminated in both terms. No pressure term or numerical stiffness is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

from ._ge_beam3_mixed_ad import Jet2, sqrt
from .e4_pl_enriched import EnrichedProfileError, bubble_basis, compile_cell, readonly
from .e4_pl_element import (
    _coefficients, _jacobian, _source_fields, _solve_stationary_system,
    _stationary_backward_error, _STATIONARY_BACKWARD_ERROR_LIMIT,
)

FORMULATION = 'E4PL_B4_CURRENT_NORMAL_ELASTIC_NONLINEAR_AFFINE_OFFSET_V4'


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Jet2.constant(0., 28))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


@dataclass(frozen=True)
class NonlinearEnrichedCell:
    """Exact immutable compiled43-field definition and physical quadrature."""
    linear: object
    inverse: np.ndarray
    quadrature: tuple
    identity: str

    def _loading(self, coordinates, *, zero_beta=False):
        q = readonly(coordinates, (28,))
        x = [Jet2.variable(v, i, 28) for i, v in enumerate(q)]
        if zero_beta:
            x[24:] = [Jet2.constant(0., 28) for _ in range(4)]
        reference, frame = self.linear.reference, self.linear.frame
        points = [[reference[i,k]+x[6*i+k] for k in range(3)] for i in range(4)]
        e1 = [points[1][k]-points[0][k] for k in range(3)]
        e2 = [points[3][k]-points[0][k] for k in range(3)]
        normal = _cross(e1, e2)
        norm_values=np.array([v.value for v in normal])
        norm_value=float(np.linalg.norm(norm_values))
        if not np.all(np.isfinite(norm_values)) or not np.isfinite(norm_value) or norm_value<=np.finfo(float).tiny:
            raise EnrichedProfileError('collapsed/nonfinite current nonlinear source frame')
        length = sqrt(_dot(normal, normal))
        if not np.isfinite(length.value) or length.value <= np.finfo(float).tiny:
            raise EnrichedProfileError('collapsed current nonlinear source frame')
        normal = [v/length for v in normal]
        if sum(normal[k].value*frame[k,2] for k in range(3)) <= 0:
            raise EnrichedProfileError('inverted current nonlinear source frame')
        local_normal = [sum((frame[k,j]*normal[k] for k in range(3)), Jet2.constant(0.,28)) for j in range(3)]
        local_u = [[sum((frame[k,j]*x[6*i+k] for k in range(3)), Jet2.constant(0.,28)) for j in range(3)] for i in range(4)]
        # The original linear load is global-coordinate transformed already.
        f = [sum((self.linear.coupling[i,j]*x[i] for i in range(28)), Jet2.constant(0.,28)) for j in range(43)]
        for dx, dy, bx, by, stress, weight in self.quadrature:
            wx = sum((dx[i]*local_u[i][2] for i in range(4)), Jet2.constant(0.,28))
            wy = sum((dy[i]*local_u[i][2] for i in range(4)), Jet2.constant(0.,28))
            beta_x = sum((bx[i]*x[24+i] for i in range(4)), Jet2.constant(0.,28))
            beta_y = sum((by[i]*x[24+i] for i in range(4)), Jet2.constant(0.,28))
            total_x, total_y = wx+beta_x*local_normal[2], wy+beta_y*local_normal[2]
            # Difference from the EXACT reference-linear compatible field.
            # Includes tilted-normal in-plane motion, total VK slopes, and
            # current-normal transverse shear, with complete normal chain.
            delta = [beta_x*local_normal[0]+total_x*total_x/2,
                     beta_y*local_normal[1]+total_y*total_y/2,
                     beta_y*local_normal[0]+beta_x*local_normal[1]+total_x*total_y,
                     Jet2.constant(0.,28),Jet2.constant(0.,28),Jet2.constant(0.,28),
                     beta_x*(local_normal[2]-1),beta_y*(local_normal[2]-1)]
            for j in (*range(14), *range(35,39)):
                f[j] = f[j]+weight*sum((stress[k,j]*delta[k] for k in range(8)), Jet2.constant(0.,28))
        values = np.array([v.value for v in f])
        gradient = np.array([v.gradient for v in f])
        hessian = np.array([v.hessian for v in f])
        if not all(np.all(np.isfinite(v)) for v in (values,gradient,hessian)):
            raise EnrichedProfileError('nonfinite nonlinear stationary loading')
        return values, gradient, hessian

    def potential(self, coordinates, *, zero_beta=False):
        """Value, complete gradient/Hessian and all43 stationary amplitudes."""
        f, df, ddf = self._loading(coordinates, zero_beta=zero_beta)
        solved = self.inverse @ f
        gradient = -df.T @ solved
        hessian = -df.T @ self.inverse @ df-np.einsum('a,aij->ij', solved, ddf)
        value = -float(f @ solved)/2
        if not all(np.all(np.isfinite(v))for v in (value,gradient,hessian,solved)):
            raise EnrichedProfileError('nonfinite nonlinear stationary potential')
        return value, gradient, hessian, -solved

    def response(self, coordinates):
        """One potential difference; no force/tangent splice or beta removal."""
        q = readonly(coordinates, (28,))
        full = self.potential(q)
        if np.all(q[24:]==0):
            # Restriction of the SAME potential to beta=0. The nodal block is
            # identical; only beta derivatives are suppressed. This avoids a
            # second loading evaluation without omitting the beta cross terms.
            force=full[1].copy();hessian=full[2].copy()
            force[:24]=0.;hessian[:24,:24]=0.
            return 0.,force,hessian
        base = self.potential(q, zero_beta=True)
        made = (full[0]-base[0], full[1]-base[1], full[2]-base[2])
        if not all(np.all(np.isfinite(v)) for v in made):
            raise EnrichedProfileError('nonfinite nonlinear enrichment response')
        return made

    def recover(self, coordinates):
        return readonly(self.potential(coordinates)[3], (43,))

    def recover_fields(self,coordinates,r,s):
        if not all(np.isfinite(x)and -1<=x<=1 for x in (r,s)):
            raise EnrichedProfileError('nonlinear recovery point outside reference Q4')
        local=(self.linear.reference-self.linear.reference.mean(axis=0)) @ self.linear.frame[:,:2]
        c=_coefficients(local);ns,ne=_source_fields(c,r,s)
        _,br,bs=bubble_basis(r,s);xr,xs,yr,ys,det=_jacobian(c,r,s)
        extra=np.zeros((8,4));extra[6]=(ys*np.asarray(br)-yr*np.asarray(bs))/det
        extra[7]=(-xs*np.asarray(br)+xr*np.asarray(bs))/det
        a=self.recover(coordinates)
        return dict(profile=FORMULATION,identity=self.identity,frame=self.linear.frame,
            stationary_amplitudes=a,generalized_stress=readonly(ns @ a[:14]+extra @ a[35:39]),
            generalized_strain=readonly(ne @ a[14:35]+extra @ a[39:43]))


def compile_nonlinear_cell(element, mesh, material):
    linear = compile_cell(element, mesh, material)
    local = (linear.reference-linear.reference.mean(axis=0)) @ linear.frame[:,:2]
    c = _coefficients(local)
    points, weights = np.polynomial.legendre.leggauss(5)
    quadrature = []
    for r, wr in zip(points,weights):
        for s, ws in zip(points,weights):
            nr = np.array((-(1-s),1-s,1+s,-(1+s)))/4
            ns = np.array((-(1-r),-(1+r),1+r,1-r))/4
            _,br,bs = bubble_basis(r,s)
            xr,xs,yr,ys,det = _jacobian(c,r,s)
            dx,dy = (ys*nr-yr*ns)/det,(-xs*nr+xr*ns)/det
            bx,by = (ys*np.array(br)-yr*np.array(bs))/det,(-xs*np.array(br)+xr*np.array(bs))/det
            stress = np.zeros((8,43))
            stress[:,:14] = _source_fields(c,r,s)[0]
            stress[6,35:39],stress[7,35:39] = bx,by
            quadrature.append(tuple(readonly(x) for x in (dx,dy,bx,by,stress))+(float(wr*ws*det),))
    inverse,_ = _solve_stationary_system(linear.stationary,np.eye(43))
    inverse = (inverse+inverse.T)/2
    backward=_stationary_backward_error(linear.stationary,inverse,np.eye(43))
    if not np.isfinite(backward) or backward>_STATIONARY_BACKWARD_ERROR_LIMIT:
        raise EnrichedProfileError('nonlinear stationary inverse lacks original backward-error certificate')
    identity = hashlib.sha256(json.dumps(dict(profile=FORMULATION,linear=linear.identity,
        law='CURRENT_NODAL_PLUS_PSI43_DIFFERENCE',quadrature='5x5',
        inverse=hashlib.sha256(inverse.tobytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
    return NonlinearEnrichedCell(linear,readonly(inverse,(43,43)),tuple(quadrature),identity)
