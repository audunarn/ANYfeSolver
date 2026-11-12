"""Private finite area-pose candidate; no owner or public admission.

The physical mean director is independent of the geometric normal. Its plane's
proper polar fit of the material gradients supplies yaw. Fixed-size macro AD
and sparse nodal pullback include multiplier curvature and spatial-row correction.
This is a distributing pose, not a pointwise welded or rigid region.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from math import fsum

import numpy as np
from scipy import sparse

from ._ge_beam3_mixed_ad import (
    Jet2, constant_matrix, dot, matmul, matvec, so3_exp, so3_log, sqrt, transpose,
    proper_rotation,
)
from ._ge_beam3_p5.algebra import skew
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._ge_beam3_p5_seeded.core import canonical
from ._stiffened_shell_distributing_footprint import CompiledDistributingFootprint
from .control import cancellation_safe_point


LAW = "GE_FLAT_Q4_DISTRIBUTING_DIRECTOR_PLANAR_POLAR_POSE_V1_CANDIDATE"
PARTITION_MARGIN = 64*np.finfo(float).eps
PROJECTED_RANK_MARGIN = np.sqrt(np.finfo(float).eps)


class DistributingPoseDomainError(ValueError):
    """An explicitly identified candidate configuration/response is inadmissible."""


def _owned(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _array(value, shape):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("complete finite binary64 physical pose arrays required")
    return _owned(value)


def _frames(value, count):
    value = _array(value, (count, 3, 3))
    for rotation in value:
        proper_rotation(rotation, label="physical shell director frame")
    return value


def _frozen_sparse(value):
    value = sparse.csr_matrix(value, copy=True)
    value.sum_duplicates(); value.eliminate_zeros(); value.sort_indices()
    value.data, value.indices, value.indptr = map(_owned, (value.data, value.indices, value.indptr))
    return value


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _unit(vector):
    scale = max(abs(v.value) for v in vector)
    if not np.isfinite(scale) or scale == 0.:
        raise DistributingPoseDomainError("representable nonzero pose vector required")
    inverse = 1./scale
    if not np.isfinite(inverse):
        raise DistributingPoseDomainError("unrepresentable pose normalization derivative")
    # A frozen scalar multiplier avoids evaluating reciprocal derivatives of
    # an already constant scale (the existing Jet2 division computes powers).
    scaled = [v*inverse for v in vector]
    length = sqrt(dot(scaled, scaled))
    return [v/length for v in scaled]


def _orientation(director, a, b):
    n = _unit(director)
    projected = [v-n[i]*dot(n, a) for i, v in enumerate(a)]
    turned = _cross(b, n)
    polar = [u+v for u, v in zip(projected, turned)]
    # Pn is idempotent analytically; reevaluate that identity on the small
    # polar vector so cancellation of large normal parts cannot spoil SO(3).
    q1 = _unit([v-n[i]*dot(n,polar) for i,v in enumerate(polar)])
    q2 = _cross(n, q1)
    return [[q1[i], q2[i], n[i]] for i in range(3)]


@dataclass(frozen=True)
class DistributingPoseEvaluation:
    identity: str
    input_sha256: str
    potential: float
    constraints: np.ndarray
    constraint_jacobian: np.ndarray
    residual: np.ndarray
    chart_hessian: sparse.csr_matrix
    spatial_jacobian: sparse.csr_matrix
    shell_position: np.ndarray
    shell_position_low: np.ndarray
    shell_frame: np.ndarray
    production_qualified: bool = field(default=False, init=False)
    owner_binding_authorized: bool = field(default=False, init=False)


class FiniteDistributingPose:
    """A local candidate evaluated from supplied physical configurations.

    Rows are spatial resisting force/moment covectors. ``chart_hessian`` is
    the symmetric Hessian in Exp perturbations at this current configuration;
    ``spatial_jacobian`` differentiates the spatial rows under those updates.
    An additive shell chart still needs the existing owner's column pullback.
    No element, accepted state, retained variable, history or checkpoint is owned.
    """
    __slots__ = ("reference_identity", "identity", "node_ids", "beam_node_id",
                 "_weights", "_gradients", "_anchor", "_offset", "_relative",
                 "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("finite distributing pose definition is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, reference):
        if type(reference) is not CompiledDistributingFootprint:
            raise ValueError("compiled Cartesian reference footprint required")
        self.reference_identity = reference.identity
        self.node_ids, self.beam_node_id = reference.shell_node_ids, reference.beam_node_id
        self._anchor = int(np.argmax(reference.area_mean))
        weights, gradients = reference.area_mean.copy(), reference.planar_gradients.copy()
        rest = [i for i in range(len(weights)) if i != self._anchor]
        # Anchored summation uses the exact partition identities. The implied
        # anchor coefficients differ from stored weights only by binary64 roundoff.
        weights[self._anchor] = 1.-fsum(float(weights[i]) for i in rest)
        for g in gradients:
            g[self._anchor] = -fsum(float(g[i]) for i in rest)
        if not np.isfinite(weights).all() or np.any(weights < 0.) or weights[self._anchor] <= 0.:
            raise ValueError("representable positive area partition required")
        self._weights, self._gradients = _owned(weights), _owned(gradients)
        position, low, _, a, b = self._aggregate(reference.shell_positions_m,
            np.zeros_like(reference.shell_positions_m), np.tile(np.eye(3), (len(weights), 1, 1)))
        frame = self._admitted_frame(np.array((0., 0., 1.)), a, b)
        self._offset = _owned(frame.T@np.array([fsum((float(reference.beam_position_m[i]),
            -float(position[i]), -float(low[i]))) for i in range(3)]))
        self._relative = _owned(frame.T)  # admitted reference beam frame is identity
        self.identity = sha256(canonical(dict(law=LAW, reference=self.reference_identity,
            nodes=self.node_ids, beam=self.beam_node_id, weights=self._weights,
            gradients=self._gradients, offset=self._offset, relative=self._relative,
            partition_margin=PARTITION_MARGIN, projected_rank_margin=PROJECTED_RANK_MARGIN))).hexdigest()
        self._sealed = True

    def descriptor(self):
        count = 6*(len(self.node_ids)+1)
        return dict(law=LAW, reference_identity=self.reference_identity, identity=self.identity,
            physical_nodes=len(self.node_ids)+1, rows=6, macro_ad_variables=18,
            maximum_local_curvature_nnz=count*count,
            update="SPATIAL_LEFT_ROTATIONS_ADDITIVE_PHYSICAL_POSITIONS",
            row_representation="SPATIAL_RESISTING_WRENCHES", relative_log_limit_rad=.9*np.pi,
            mean_director_margin=PARTITION_MARGIN, projected_rank_margin=PROJECTED_RANK_MARGIN,
            public_admission=False, owner_binding_authorized=False, production_qualified=False)

    def _aggregate(self, positions, low, frames):
        count, anchor = len(self.node_ids), self._anchor
        delta = np.array([[fsum((float(positions[i,j]), -float(positions[anchor,j]),
            float(low[i,j]), -float(low[anchor,j]))) for j in range(3)] for i in range(count)])
        mean_high, mean_low = np.empty(3), np.empty(3)
        for j in range(3):
            mean_high[j], mean_low[j] = split_sum((float(positions[anchor,j]), float(low[anchor,j]),
                *(float(self._weights[i])*float(delta[i,j]) for i in range(count))))
        director = np.array([fsum(float(w)*float(q[j,2]) for w,q in zip(self._weights,frames)) for j in range(3)])
        tangents = np.array([[fsum(float(g[i])*float(delta[i,j]) for i in range(count))
                             for j in range(3)] for g in self._gradients])
        if not all(np.isfinite(v).all() for v in (mean_high,mean_low,director,tangents)):
            raise DistributingPoseDomainError("unrepresentable aggregate pose")
        return mean_high, mean_low, director, *tangents

    @staticmethod
    def _admitted_frame(director, a, b):
        length = float(np.linalg.norm(director))
        if not np.isfinite(length) or length <= PARTITION_MARGIN:
            raise DistributingPoseDomainError("mean physical directors cancel")
        n = director/length
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if not np.isfinite(scale) or scale == 0.:
            raise DistributingPoseDomainError("collapsed projected material gradients")
        # This bounded 3x2 kinematic rank test makes no structural SPD assumption.
        B = np.column_stack((a/scale,b/scale))
        T = B-n[:,None]*(n@B)
        singular = np.linalg.svd(T, compute_uv=False)
        # Both projected columns can be tiny relative to their normal parts.
        # Include the unprojected magnitude, not just T's condition number.
        if singular[-1] <= PROJECTED_RANK_MARGIN*max(singular[0],float(np.linalg.norm(B))):
            raise DistributingPoseDomainError("ill-conditioned projected material gradients")
        # Test orientation after projection, with a product/summation roundoff
        # bound; raw B's large normal parts can hide the oriented tangent area.
        u,v=T[:,0],T[:,1]
        terms=(n[0]*u[1]*v[2],-n[0]*u[2]*v[1],n[1]*u[2]*v[0],
               -n[1]*u[0]*v[2],n[2]*u[0]*v[1],-n[2]*u[1]*v[0])
        gamma=32*np.finfo(float).eps/(1-32*np.finfo(float).eps)
        if fsum(float(t) for t in terms) <= gamma*fsum(abs(float(t)) for t in terms):
            raise DistributingPoseDomainError("reversed material-gradient orientation")
        v = T[:,0]+np.cross(T[:,1],n)
        v = v-n*float(n@v)
        q1 = v/np.linalg.norm(v)
        frame = np.column_stack((q1,np.cross(n,q1),n))
        return proper_rotation(frame, label="distributing polar frame")

    def evaluate(self, shell_positions, shell_frames, beam_position, beam_frame, multipliers,
                 *, shell_position_low=None, beam_position_low=None, cancellation_token=None):
        cancellation_safe_point(cancellation_token, "distributing-pose.start")
        n = len(self.node_ids); size = 6*(n+1)
        x = _array(shell_positions, (n,3)); Q = _frames(shell_frames,n)
        rb = _array(beam_position,(3,)); Qb = proper_rotation(_array(beam_frame,(3,3)),label="physical beam frame")
        force = _array(multipliers,(6,))
        low = np.zeros_like(x) if shell_position_low is None else _array(shell_position_low,x.shape)
        bl = np.zeros(3) if beam_position_low is None else _array(beam_position_low,(3,))
        try:
            rs, rslow, director, a, b = self._aggregate(x,low,Q)
            gap = np.array([fsum((float(rb[i]), -float(rs[i]), float(bl[i]), -float(rslow[i]))) for i in range(3)])
        except OverflowError as error:
            raise DistributingPoseDomainError("unrepresentable aggregate pose or port gap") from error
        self._admitted_frame(director,a,b)
        values = np.r_[np.zeros(3),director,a,b,gap,np.zeros(3)]
        z = [Jet2.variable(float(value),i,18) for i,value in enumerate(values)]
        made = _orientation(z[3:6],z[6:9],z[9:12])
        frame = proper_rotation(np.array([[v.value for v in row] for row in made]),label="evaluated distributing polar frame")
        beam = matmul(so3_exp(z[15:18]),constant_matrix(Qb,18))
        ct = [v-float(e) for v,e in zip(matvec(transpose(made),[z[12+i]-z[i] for i in range(3)]),self._offset)]
        cr = so3_log(matmul(constant_matrix(self._relative.T,18),matmul(transpose(made),beam)))
        c = ct+cr
        work = sum((v*float(lam) for v,lam in zip(c,force)),Jet2.constant(0.,18))
        # Macro director second variations belong to each nodal Exp chart.
        D = sparse.lil_matrix((18,size))
        for i,w in enumerate(self._weights):
            for j in range(3):
                D[j,6*i+j] = w
                D[6+j,6*i+j] = self._gradients[0,i]
                D[9+j,6*i+j] = self._gradients[1,i]
            D[3:6,6*i+3:6*i+6] = -w*skew(Q[i,:,2])
        D[12:15,6*n:6*n+3] = np.eye(3); D[15:18,6*n+3:size] = np.eye(3)
        D = D.tocsr()
        residual = np.asarray(D.T@work.gradient).ravel()
        hessian = D.T@sparse.csr_matrix(work.hessian)@D
        extra = sparse.lil_matrix((size,size))
        for i,w in enumerate(self._weights):
            d, v = Q[i,:,2],work.gradient[3:6]
            extra[6*i+3:6*i+6,6*i+3:6*i+6] = w*(.5*(np.outer(d,v)+np.outer(v,d))-float(d@v)*np.eye(3))
        hessian = (hessian+extra.tocsr()).tocsr()
        correction = sparse.lil_matrix((size,size))
        for i in range(n+1):
            start = 6*i+3
            correction[start:start+3,start:start+3] = -.5*skew(residual[start:start+3])
        spatial = (hessian+correction.tocsr()).tocsr()
        jacobian = (sparse.csr_matrix([v.gradient for v in c])@D).toarray()
        constraints = np.array([v.value for v in c])
        if not np.isfinite(work.value) or not all(np.isfinite(v).all() for v in
            (constraints,jacobian,residual,hessian.data,spatial.data)):
            raise DistributingPoseDomainError("unrepresentable finite distributing work derivative")
        fingerprint = sha256(canonical(dict(positions=x,position_low=low,frames=Q,
            beam_position=rb,beam_low=bl,beam_frame=Qb,multipliers=force))).hexdigest()
        result = DistributingPoseEvaluation(self.identity,fingerprint,float(work.value),
            _owned(constraints),_owned(jacobian),_owned(residual),_frozen_sparse(hessian),
            _frozen_sparse(spatial),_owned(rs),_owned(rslow),_owned(frame))
        cancellation_safe_point(cancellation_token, "distributing-pose.complete")
        return result
