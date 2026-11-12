"""Nodal physical director map for explicitly admitted corotational owners.

The physical field is Exp([theta]x)d, rather than an element frame dependent
linearized director. Forces use spatial virtual spins; derivatives below use
the solver's additive rotation coordinates. These two maps are distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse

from .q4_s3_overlap_derivatives import _BJet

_TERMS = 32
_EPS = np.finfo(float).eps


def _up(x):
    return np.nextafter(np.asarray(x, dtype=float), np.inf)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _skew(a):
    x, y, z = a
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def director_jets(theta, reference_directors):
    """Same Exp field as the structural owner, with an enclosed Taylor value.

    A batched existing Jet2 algebra supplies *both* nonzero variations at zero.
    In particular the owner's tiny-angle value approximation is not used to
    erase the second variation of the adopted physical exponential field.
    """
    theta = np.asarray(theta, dtype=float)
    dref = np.asarray(reference_directors, dtype=float)
    if (theta.ndim != 2 or theta.shape[1] != 3 or dref.shape != theta.shape
            or not np.isfinite(theta).all() or not np.isfinite(dref).all()):
        raise ValueError('finite director needs finite matching (n,3) arrays')
    count = len(theta)
    variables = []
    for axis in range(3):
        g = np.zeros((count, 3)); g[:, axis] = 1.
        variables.append(_BJet(theta[:, axis], g, np.zeros((count, 3, 3))))
    term = tuple(_BJet.const(dref[:, j], 3) for j in range(3))
    result = term
    for n in range(1, _TERMS+1):
        term = tuple(x.scale(1./n) for x in _cross(variables, term))
        result = tuple(a+b for a, b in zip(result, term))
    value = np.column_stack([x.v for x in result])
    gradient = np.stack([x.g for x in result], axis=1)
    hessian = np.stack([x.h for x in result], axis=1)
    return value, gradient, hessian


def director_value_error(theta, reference_directors):
    """Absolute component enclosure, without a libm sine/cosine assumption.

    Each Taylor monomial path contains fewer than 512 rounded operations
    (32 cross recurrences, divisions and summation). The induced infinity
    skew norm is at most |theta|_1. A gamma512 absolute monomial majorant
    encloses recurrence rounding, including rounded 1/n coefficients. The
    exponential integral remainder is <= M^33/33! |dref|_1. Subnormal absolute
    losses are included independently. All positive arithmetic is outward.
    """
    theta = np.asarray(theta, dtype=float)
    dref = np.asarray(reference_directors, dtype=float)
    m = _up(_up(np.abs(theta[:, 0])+np.abs(theta[:, 1]))
            +np.abs(theta[:, 2]))
    return _director_error_from_l1(m, dref)


def _director_error_from_l1(m, dref):
    d = _up(_up(np.abs(dref[:, 0])+np.abs(dref[:, 1]))
            +np.abs(dref[:, 2]))
    term = np.ones(len(m)); total = term.copy()
    for n in range(1, _TERMS+1):
        term = _up(_up(term*m)/n)
        total = _up(total+term)
    tail = _up(_up(term*m)/(_TERMS+1))
    gamma = np.nextafter(512*_EPS/(1.-512*_EPS), np.inf)
    return _up(_up(_up(gamma*total)+tail)*d
               +512*np.nextafter(0., 1.))


def uniform_skin_evaluation_error(skin, qa, qb, ea, eb):
    """Uniform numerical-map enclosure on a whole affine coordinate interval."""
    a = qa[skin.dofs]; b = qb[skin.dofs]
    abs_upper = _up(np.maximum(np.abs(a)+ea[skin.dofs],
                              np.abs(b)+eb[skin.dofs]))
    m = _up(_up(abs_upper[:, 3]+abs_upper[:, 4])+abs_upper[:, 5])
    derr = _director_error_from_l1(m, skin.directors)
    dnorm = director_norm_upper(skin.directors)
    director_value = _up(dnorm+derr)
    magnitude = _up(_up(np.abs(skin.reference)+abs_upper[:, :3])
                    +_up(np.abs(skin.offsets[:, None])*director_value[:, None]))
    eps = np.finfo(float).eps
    gamma = np.nextafter(3*eps/(1.-3*eps), np.inf)
    return _up(_up(np.abs(skin.offsets[:, None])*derr[:, None])
               +_up(gamma*magnitude)+3*np.nextafter(0., 1.))


def director_norm_upper(dref):
    """Upper Euclidean norm of persisted binary directors, not nominal one."""
    dref = np.asarray(dref, dtype=float)
    squared = _up(_up(_up(dref[:, 0]*dref[:, 0])
                      +_up(dref[:, 1]*dref[:, 1]))
                  +_up(dref[:, 2]*dref[:, 2]))
    # sqrt is corrected by checking its square with a downward enclosure.
    root = _up(np.sqrt(squared))
    while np.any(np.nextafter(root*root, -np.inf) < squared):
        root = _up(root)
    return root


def curvature_radius(theta_start, theta_end, offsets, dref, *,
                     coordinate_errors=None):
    """Whole affine-q subinterval Exp curve deviation from its endpoint chord.

    The second Frechet derivative simplex formula gives ||Q''||<=||delta||^2.
    Thus the exact curve remainder is <= |h| ||dref|| ||delta||^2/8. Input
    q endpoint rounding boxes also contribute to delta's directed upper bound.
    Evaluation/skin errors are added by the caller, separately from curvature.
    """
    a = np.asarray(theta_start, float); b = np.asarray(theta_end, float)
    delta = _up(np.abs(b-a))
    subtraction = _up(_EPS*_up(np.abs(a)+np.abs(b)))
    delta = _up(delta+subtraction)
    if coordinate_errors is not None:
        e0, e1 = coordinate_errors
        delta = _up(delta+_up(np.asarray(e0)+np.asarray(e1)))
    sq = _up(delta*delta)
    norm2 = _up(_up(sq[:, 0]+sq[:, 1])+sq[:, 2])
    return _up(_up(_up(np.abs(offsets)*director_norm_upper(dref))*norm2)/8.)


@dataclass(frozen=True)
class FiniteSkinPose:
    positions: np.ndarray
    directors: np.ndarray
    director_gradient: np.ndarray
    coordinate_map: sparse.csr_matrix
    spin_map: sparse.csr_matrix
    position_errors: np.ndarray


class FiniteQ4SkinMap:
    """Node-ID aligned nonlinear skins and both work maps, with no normal flip."""
    def __init__(self, model, adapters, geometry_node_ids):
        records = {}
        for adapter in adapters:
            element = model.mesh.elements[adapter.element_id]
            original = dict(zip(element.node_ids,
                                element.get_node_coordinates(model.mesh)))
            for node in adapter.node_ids:
                record = (np.asarray(original[node], float), adapter.director,
                          adapter.physical_offset)
                if node in records:
                    old = records[node]
                    if (not np.array_equal(old[0], record[0])
                            or not np.array_equal(old[1], record[1])
                            or old[2] != record[2]):
                        raise ValueError('shared Q4 physical skin declaration differs')
                records[node] = record
        ids = tuple(geometry_node_ids)
        self.reference = np.array([records[n][0] for n in ids])
        self.directors = np.array([records[n][1] for n in ids])
        self.offsets = np.array([records[n][2] for n in ids])
        self.dofs = np.array([model.mesh.dof_manager.get_node_dofs(n)[:6]
                              for n in ids], dtype=np.intp)
        self.n_dof = model.mesh.dof_manager.total_dofs

    def pose(self, displacement):
        q = np.asarray(displacement, dtype=float).reshape(-1)
        if q.size != self.n_dof or not np.isfinite(q).all():
            raise ValueError('finite skin pose has invalid displacement')
        nodal = q[self.dofs]
        d, dg, _ = director_jets(nodal[:, 3:], self.directors)
        positions = self.reference+nodal[:, :3]+self.offsets[:, None]*d
        rows = []; cols = []; cv = []; sv = []
        for i, dofs in enumerate(self.dofs):
            spin = -self.offsets[i]*_skew(d[i])
            chart = self.offsets[i]*dg[i]
            for a in range(3):
                for b in range(6):
                    rows.append(3*i+a); cols.append(int(dofs[b]))
                    cv.append(float(a == b) if b < 3 else chart[a, b-3])
                    sv.append(float(a == b) if b < 3 else spin[a, b-3])
        shape = (3*len(d), self.n_dof)
        coordinate = sparse.coo_matrix((cv, (rows, cols)), shape=shape).tocsr()
        spin = sparse.coo_matrix((sv, (rows, cols)), shape=shape).tocsr()
        err = director_value_error(nodal[:, 3:], self.directors)
        magnitude = _up(_up(np.abs(self.reference)+np.abs(nodal[:, :3]))
                        +_up(np.abs(self.offsets[:, None]*d)))
        gamma = np.nextafter(3*_EPS/(1.-3*_EPS), np.inf)
        errors = _up(_up(np.abs(self.offsets[:, None])*err[:, None])
                     +_up(gamma*magnitude))
        return FiniteSkinPose(positions, d, dg, coordinate, spin, errors)

    def spin_force_derivative(self, pose, local_force):
        """Derivative of h*d cross f with f held fixed, in additive q charts."""
        force = np.asarray(local_force, float).reshape(-1, 3)
        result = sparse.lil_matrix((self.n_dof, self.n_dof))
        for i, dofs in enumerate(self.dofs):
            block = -self.offsets[i]*_skew(force[i])@pose.director_gradient[i]
            result[np.ix_(dofs[3:], dofs[3:])] += block
        return result.tocsr()
