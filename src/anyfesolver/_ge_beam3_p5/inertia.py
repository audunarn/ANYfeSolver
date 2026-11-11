"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_p5.algebra import HALVES, skew, validate_section
from anyfesolver._ge_beam3_p5.arrays import _array, _frames, _readonly


SCHEMA = 'GE_BEAM3_P5_FINITE_RETAINED_INERTIA_DEVELOPMENT_V1'


@dataclass(frozen=True)
class FiniteInertiaResponse:
    kinetic_energy: float
    mass: np.ndarray
    generalized_momentum: np.ndarray
    inertia: np.ndarray
    velocity_derivative: np.ndarray
    configuration_derivative: np.ndarray
    linear_momentum: np.ndarray
    angular_momentum: np.ndarray


class LiftedInertia:
    def __init__(self, reference, section_mass, *, order=24):
        if type(order) is not int or not 2 <= order <= 64:
            raise ValueError('bounded integer kinetic quadrature order required')
        self.section_mass = _readonly(validate_section(section_mass))
        nodes = _array(reference.coordinates, (3,3), 'reference coordinates')
        points, weights = np.polynomial.legendre.leggauss(order)
        stations = []
        for cell,(left,right) in enumerate(HALVES):
            for point,weight in zip(points,weights):
                t = (point+1)/2;xi = cell-1+t
                frame = _frames([reference.frame(xi)],1,'reference station frame')[0]
                offset = _array(reference.position(xi),(3,),'reference position')-((1-t)*nodes[left]+t*nodes[right])
                measure = float(weight*reference.jacobian(xi)/2)
                if not np.isfinite(measure) or measure<=0.: raise ValueError('positive finite kinetic measure required')
                stations.append((cell,left,right,float(t),_readonly(frame),_readonly(offset),measure))
        self._stations = tuple(stations)

    def evaluate(self, positions, local_rotations, velocity, acceleration):
        """Spatial velocity/acceleration in external18 + cell6 ordering.

        Configuration derivatives use left multiplicative cell increments,
        holding the supplied spatial velocities and accelerations fixed. The
        velocity derivative includes centrifugal and momentum cross terms.
        These are force derivatives, not a symmetric potential Hessian.
        """
        x = _array(positions,(3,3),'nodal positions')
        cells = _frames(local_rotations,2,'cell rotations')
        speed = _array(velocity,(24,),'spatial velocity')
        accel = _array(acceleration,(24,),'spatial acceleration')
        mass = np.zeros((24,24));inertia = np.zeros(24);momentum = np.zeros(24)
        dv = np.zeros_like(mass);dq = np.zeros_like(mass)
        linear = np.zeros(3);angular = np.zeros(3);energy = 0.
        j = self.section_mass
        for cell,left,right,t,r0,d0,measure in self._stations:
            slot = slice(18+3*cell,21+3*cell)
            q = cells[cell]@r0;d = cells[cell]@d0
            b = np.zeros((6,24))
            b[:3,6*left:6*left+3] = (1-t)*np.eye(3)
            b[:3,6*right:6*right+3] = t*np.eye(3)
            b[:3,slot] = -skew(d);b[3:,slot] = np.eye(3)
            transform = np.kron(np.eye(2),q)
            body_map = transform.T@b
            field = b@speed;v,omega = field[:3],field[3:]
            acceleration_field = b@accel
            alpha = acceleration_field[3:]
            a = acceleration_field[:3]+np.cross(omega,np.cross(omega,d))
            u,w = q.T@v,q.T@omega
            body_velocity = np.r_[u,w]
            body_acceleration = np.r_[q.T@a-np.cross(w,u),q.T@alpha]
            p,h = np.split(j@body_velocity,2)
            force_body,moment_body = np.split(j@body_acceleration,2)
            force_body += np.cross(w,p)
            moment_body += np.cross(w,h)+np.cross(u,p)
            f,n = q@force_body,q@moment_body
            station_force = np.r_[f,n]
            energy += measure*float(body_velocity@(j@body_velocity))/2
            mass += measure*(body_map.T@j@body_map)
            momentum += measure*(body_map.T@(j@body_velocity))
            inertia += measure*(b.T@station_force)
            linear += measure*(q@p)
            angular += measure*(np.cross((1-t)*x[left]+t*x[right]+d,q@p)+q@h)

            def derivative(eta,dd,delta_v,delta_omega,delta_a,delta_alpha):
                # Exact product rule in spatial variables; no frame differences.
                du = q.T@(delta_v-np.cross(eta,v))
                dw = q.T@(delta_omega-np.cross(eta,omega))
                da = q.T@(delta_a-np.cross(eta,a))-np.cross(dw,u)-np.cross(w,du)
                dalpha = q.T@(delta_alpha-np.cross(eta,alpha))
                dp,dh = np.split(j@np.r_[du,dw],2)
                df,dn = np.split(j@np.r_[da,dalpha],2)
                df += np.cross(dw,p)+np.cross(w,dp)
                dn += np.cross(dw,h)+np.cross(w,dh)+np.cross(du,p)+np.cross(u,dp)
                changed = b.T@np.r_[q@df+np.cross(eta,f),q@dn+np.cross(eta,n)]
                changed[slot] += np.cross(dd,f)
                return measure*changed

            zero = np.zeros(3)
            for column in np.flatnonzero(np.any(b!=0.,axis=0)):
                change = b[:,column];delta_v,delta_omega = change[:3],change[3:]
                delta_a = np.cross(delta_omega,np.cross(omega,d))+np.cross(omega,np.cross(delta_omega,d))
                dv[:,column] += derivative(zero,zero,delta_v,delta_omega,delta_a,zero)
            for k,eta in enumerate(np.eye(3)):
                dd = np.cross(eta,d)
                delta_v = np.cross(omega,dd)
                delta_a = np.cross(alpha,dd)+np.cross(omega,np.cross(omega,dd))
                dq[:,18+3*cell+k] += derivative(eta,dd,delta_v,zero,delta_a,zero)
        arrays = (mass,momentum,inertia,dv,dq,linear,angular)
        if not np.isfinite(energy) or energy<0. or any(not np.isfinite(a).all() for a in arrays):
            raise ValueError('nonfinite kinetic response')
        return FiniteInertiaResponse(float(energy),*(_readonly(a) for a in arrays))
