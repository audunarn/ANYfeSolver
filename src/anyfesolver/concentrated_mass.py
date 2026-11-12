"""Physical rigid mass attached to a six-DOF node; all quantities in model axes/SI.

Restriction: every mesh node carries six DOFs, so admission only confirms that
the attachment node exists. It does not assess whether the node's rotations are
restrained by elements (beam/shell) or constraints. Rotary inertia on a node
whose rotations carry no stiffness is admitted unchanged; free rigid motion is
legitimate for inertia relief and modal use, and no stiffness threshold is
imposed here. Callers attaching to translational-only (e.g. solid) nodes are
responsible for providing rotational coupling.
"""
from dataclasses import dataclass
import numpy as np

# Rotating a physical tensor (R J R^T) leaves antisymmetric roundoff of a few
# ulps of its largest entry; anything larger is treated as nonsymmetric input.
_SYMMETRY_RTOL = 64.0 * np.finfo(float).eps


@dataclass(frozen=True)
class ConcentratedMass:
    mass: float
    offset: tuple = (0., 0., 0.)
    inertia_cg: tuple = ((0.,0.,0.),(0.,0.,0.),(0.,0.,0.))

    def __post_init__(self):
        m = float(self.mass)
        r = np.asarray(self.offset, dtype=float)
        J = np.asarray(self.inertia_cg, dtype=float)
        if not np.isfinite(m) or m < 0 or r.shape != (3,) or not np.all(np.isfinite(r)):
            raise ValueError('concentrated mass and offset must be finite; mass must be nonnegative')
        if J.shape != (3,3) or not np.all(np.isfinite(J)):
            raise ValueError('inertia_cg must be a finite symmetric 3x3 tensor')
        scale = float(np.max(np.abs(J)))
        if float(np.max(np.abs(J-J.T))) > _SYMMETRY_RTOL*scale:
            raise ValueError('inertia_cg must be a finite symmetric 3x3 tensor')
        J = 0.5*(J+J.T)
        e = np.linalg.eigvalsh(J)
        tol = 1e-12 * max(float(np.max(np.abs(e))), np.finfo(float).tiny)
        if e[0] < -tol or e[2] > e[0]+e[1]+tol or (m == 0 and np.any(J)):
            raise ValueError('inertia must be physical: nonnegative principal moments, triangle inequalities, positive mass')
        object.__setattr__(self,'mass',m)
        object.__setattr__(self,'offset',tuple(float(v) for v in r))
        object.__setattr__(self,'inertia_cg',tuple(tuple(float(v) for v in row) for row in J))

    def matrix(self):
        x,y,z=self.offset
        S=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        B=np.eye(6); B[:3,3:]=-S
        D=np.zeros((6,6)); D[:3,:3]=self.mass*np.eye(3); D[3:,3:]=self.inertia_cg
        return B.T@D@B


def entries(mesh):
    masses=getattr(mesh,'concentrated_masses',{})
    if type(masses) is not dict:
        raise ValueError('concentrated_masses must be a dict of node id to ConcentratedMass lists')
    for node_id, values in sorted(masses.items()):
        if type(node_id) is not int:
            raise ValueError('concentrated_masses keys must be int node ids')
        if type(values) not in (list, tuple):
            raise ValueError('concentrated_masses values must be lists of ConcentratedMass records')
        node=mesh.get_node(node_id)
        if node is None or len(node.dofs)!=6:
            raise ValueError('concentrated mass requires an existing six-DOF attachment node')
        for value in values:
            if type(value) is not ConcentratedMass:
                raise ValueError('concentrated_masses must contain ConcentratedMass records')
            # Revalidate even objects restored through transports bypassing constructors.
            checked=ConcentratedMass(value.mass,value.offset,value.inertia_cg)
            yield node,checked


def identity(mesh):
    return tuple((int(n.id),v.mass,v.offset,v.inertia_cg) for n,v in entries(mesh))


def add_to_matrix(mesh, matrix):
    from scipy import sparse
    rows=[]; cols=[]; data=[]
    for node,value in entries(mesh):
        block=value.matrix()
        rows.extend(np.repeat(node.dofs,6)); cols.extend(np.tile(node.dofs,6)); data.extend(block.ravel())
    if not data: return matrix
    return (matrix+sparse.coo_matrix((data,(rows,cols)),shape=matrix.shape)).tocsr()
