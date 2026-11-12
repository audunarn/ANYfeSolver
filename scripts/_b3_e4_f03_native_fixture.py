"""Unregistered F03 diagnostic fixture for the native retained generalized B3 chain.

The straight reference axis is x=s in [0, 1] metres. Two half-cells form
each actual three-node native B3 element. The C1 profile has explicit
station cuts and bounded bond-only subdivisions.
The law is a positive *generalized resultant* law, not a fibre surrogate:
E=210 GPa, nu=.3, G=E/(2*(1+nu)), A=.01 m², Ay=Az=.008 m²,
Iy=Iz=8.333333333333333e-6 m⁴, J=1.6666666666666667e-5 m⁴.
In [N,Vy,Vz,Mx,My,Mz] work order the diagonal elastic entries are
[EA,G Ay,G Az,G J,E Iy,E Iz].  Yield force 1e20 and hardening 1e9
are declared section parameters; all observations here are virgin elastic.

The six physical DOFs of the root node are fixed.  All other nodal DOFs and
all native internal coordinates are free.  Assembly calls the same retained
Context.assemble/F08 local spatial-Jacobian seam as the native implementation.
No registered gate or product code is called by importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from anyfesolver.boundary import BoundaryCondition
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_retained_generalized_state import Context, Program
from anyfesolver._ge_beam3_fibre_line_work import evaluate as line_work
from anyfesolver._ge_beam3_generalized_static_boundary import cell_couple_load, spatial_jacobian
from anyfesolver._ge_beam3_operator_validation_scope import operator_call


E = 210e9
NU = .3
G = E / (2 * (1 + NU))
AREA = .01
SHEAR_AREA = .008
I = 8.333333333333333e-6
J = 2 * I
SECTION_DIAGONAL = np.array([E * AREA, G * SHEAR_AREA, G * SHEAR_AREA,
                             G * J, E * I, E * I])


@dataclass
class Fixture:
    """Live native assembly, with local full-block access for a coupled KKT.

    ``tangent`` is the actual sparse virgin residual Jacobian, including retained
    rotational spatial-chart corrections. ``physical`` lists supported nodal
    DOF indices; ``internal`` lists all native cell rotations and 18 resultants
    per element.  The full tangent is a mixed saddle matrix, not an SPD energy
    norm.  ``blocks()`` returns Kpp,Kpr,Krp,Krr; callers may eliminate internal
    coordinates only while carrying the trace's Cr columns as well.
    """

    n_half_cells: int
    model: FEModel
    context: Context
    residual: np.ndarray | None
    tangent: sparse.csr_matrix | None
    physical: np.ndarray
    internal: np.ndarray
    dense_oracle_max_error: float | None
    abscissae: np.ndarray

    def blocks(self):
        k = self.tangent
        if k is None:
            raise RuntimeError('kinematic-only native fixture has no tangent blocks')
        return (k[np.ix_(self.physical, self.physical)].toarray(),
                k[np.ix_(self.physical, self.internal)].toarray(),
                k[np.ix_(self.internal, self.physical)].toarray(),
                k[np.ix_(self.internal, self.internal)].toarray())

    def _locate(self, s):
        if not np.isfinite(s) or not 0 <= s <= 1:
            raise ValueError('station s must be finite and in [0, 1]')
        half = min(int(np.searchsorted(self.abscissae, s, side="right"))-1,
                   self.n_half_cells-1)
        element, cell = divmod(half, 2)
        t = ((s-self.abscissae[half]) /
             (self.abscissae[half+1]-self.abscissae[half]))
        return element, cell, t, cell - 1 + t

    def pose(self, s, step=None):
        """Return actual native lifted station (position, spatial frame).

        ``step`` is a full retained increment with fixed physical DOFs zero;
        cell frames are updated by the native Context.advance spatial chart.
        The reference has zero quadratic lift, but the same native lift is
        evaluated here to preserve the declared pose route.
        """
        e, c, t, xi = self._locate(s)
        state = self.context.initial.mechanical
        if step is not None:
            delta = np.asarray(step, dtype=float)
            if delta.shape != (self.context.count,):
                raise ValueError('full retained step required')
            state = self.context.advance(state, delta)
        nodes = self.context.nodes[e]
        left, right = nodes[c:c+2]
        op = self.context.elements[e][1].operator
        lift = op.reference.half_cell_lift(c, t)
        rot = state.cell_rotations[e, c]
        position = ((1-t) * (state.positions[left] + state.position_low[left])
                    + t * (state.positions[right] + state.position_low[right])
                    + rot @ lift)
        return position, rot @ op.reference.frame(xi)

    def planar_trace(self, s, step=None):
        """Return [ux, uy, spatial Rz] at the station relative to virgin pose."""
        position, frame = self.pose(s, step)
        reference_position, reference_frame = self.pose(s)
        angle = Rotation.from_matrix(frame @ reference_frame.T).as_rotvec()[2]
        return np.array([*(position-reference_position)[:2], angle])

    def planar_trace_jacobian(self, s):
        """Exact virgin derivative, 3 by full retained DOFs (Cp and Cr).

        For a straight axis the native half-cell lift vanishes.  Translation
        rows interpolate adjacent native positions; Rz is the spatial cell
        rotation increment.  The selected native cell rotation column belongs
        to ``internal``, never to the nodal ``physical`` block.
        """
        e, c, t, _ = self._locate(s)
        nodes = self.context.nodes[e]
        out = np.zeros((3, self.context.count))
        out[0, 6*nodes[c]] = 1-t
        out[0, 6*nodes[c+1]] = t
        out[1, 6*nodes[c]+1] = 1-t
        out[1, 6*nodes[c+1]+1] = t
        out[2, self.context.nodal_count + 24*e + 3*c + 2] = 1
        return out

    def trace_blocks(self, s):
        trace = self.planar_trace_jacobian(s)
        return trace[:, self.physical], trace[:, self.internal]


def _sparse_native_assembly(context):
    """F08 seam: exact local operator evaluation and COO to CSR gather.

    The local 42-vector contains 18 physical DOFs, 6 cell rotations, and 18
    generalized resultants.  Its spatial Jacobian is the actual native seam,
    not an interpolated or surrogate beam stiffness.
    """
    state = context.initial.mechanical
    origins = context.initial.origins
    residual = np.zeros(context.count)
    rows, cols, data = [], [], []
    for i, (eid, element) in enumerate(context.elements):
        nodes = context.nodes[i]
        op = element.operator
        response = operator_call(
            op, 'evaluate', context.guard, context.started,
            state.positions[nodes], state.position_low[nodes], state.nodal_frames[nodes],
            state.cell_rotations[i], state.resultants[i], origin=origins[i])
        work = line_work(op.reference, state.positions[nodes], state.position_low[nodes],
                         state.cell_rotations[i], context.program.pattern.force(eid),
                         order=op.order)
        rc = response.residual - work.gradient
        local_r = rc - cell_couple_load(op, context.program.pattern.density(eid))
        local_k = spatial_jacobian(rc, response.hessian + response.hessian_low - work.hessian)
        ids = np.asarray(context.slots[i], dtype=int)
        np.add.at(residual, ids, local_r)
        rows.extend(np.repeat(ids, len(ids)))
        cols.extend(np.tile(ids, len(ids)))
        data.extend(local_k.ravel())
    tangent = sparse.coo_matrix((data, (rows, cols)),
                                shape=(context.count, context.count)).tocsr()
    tangent.eliminate_zeros()
    return residual, tangent


_C1_HALF_STATIONS = np.array((0., .125, .25, .375, .5, .625,
                              .75, .7875, .825, .8625, .9, .95, 1.))


def _c1_offset_slope(x: float) -> tuple[float, float]:
    clearance = 2e-6
    if x <= .75:
        return clearance, 0.
    if x >= .9:
        return 0., 0.
    t = (x-.75)/.15
    if t <= .5:
        return clearance*(1.-2.*t*t), -4.*clearance*t/.15
    return 2.*clearance*(1.-t)**2, -4.*clearance*(1.-t)/.15


def build(n_half_cells: int, *, z_offset: float = 0.,
          torsion_constant: float | None = None,
          reference_profile: str = "straight",
          assemble: bool = True) -> Fixture:
    """Build a native B3 chain; omit assembly for pose-only diagnostics."""
    if type(assemble) is not bool:
        raise ValueError('assemble must be a boolean')
    if (reference_profile not in ("straight", "straight_c1_stations",
                                  "c1_tangent_2um") or
            type(n_half_cells) is not int or
            (n_half_cells not in (4, 8, 16, 32) if
             reference_profile == "straight" else
             n_half_cells not in ((12, 14, 18, 24, 26) if
                                  reference_profile == "c1_tangent_2um"
                                  else (12,)))):
        raise ValueError('bounded straight or 12/14/18/24/26-half-cell C1 native reference required')
    if (isinstance(z_offset, bool) or not np.isfinite(z_offset) or
            not -.2 <= z_offset <= .2):
        raise ValueError('bounded finite native beam reference offset required')
    if torsion_constant is not None and (
            isinstance(torsion_constant, bool) or
            not np.isfinite(torsion_constant) or torsion_constant <= 0.):
        raise ValueError('positive finite physical torsion constant required')
    if reference_profile != "straight" and z_offset != .0675:
        raise ValueError('C1 stations require the matched 67.5 mm beam axis')
    n_elements = n_half_cells // 2
    if reference_profile == "straight":
        abscissae = np.arange(n_half_cells+1, dtype=float)/n_half_cells
    elif n_half_cells == 12:
        abscissae = _C1_HALF_STATIONS.copy()
    elif n_half_cells == 24:
        abscissae = np.empty(25)
        abscissae[::2] = _C1_HALF_STATIONS
        abscissae[1::2] = (_C1_HALF_STATIONS[:-1]+_C1_HALF_STATIONS[1:])/2
    elif n_half_cells in (14, 18, 26):
        bond_divisions = (n_half_cells-10)//2
        abscissae = np.r_[
            _C1_HALF_STATIONS[:10],
            np.linspace(.9, .95, bond_divisions+1)[:-1],
            np.linspace(.95, 1., bond_divisions+1),
        ]
    else:
        raise AssertionError("unreachable C1 half-cell resolution")
    name = f'f03-native-virgin-{n_half_cells}'
    if z_offset:
        name += f'-z{z_offset:.17g}'
    if reference_profile != "straight":
        name += f'-{reference_profile}'
    model = FEModel(name)
    for node, x in enumerate(abscissae):
        offset = (_c1_offset_slope(float(x))[0] if
                  reference_profile == "c1_tangent_2um" else 0.)
        model.add_node(node + 1, float(x), 0., z_offset+offset)
    section_diagonal = SECTION_DIAGONAL.copy()
    if torsion_constant is not None:
        section_diagonal[3] = G * torsion_constant
    section = EllipsoidalGeneralizedSection(np.diag(section_diagonal), np.eye(6),
                                           1e20, 1e9)
    for element in range(n_elements):
        ids = (2*element+1, 2*element+2, 2*element+3)
        coords = np.array([model.mesh.nodes[node].coords() for node in ids])
        frames = np.tile(np.eye(3), (3, 1, 1))
        if reference_profile == "c1_tangent_2um":
            for local, x in enumerate(abscissae[2*element:2*element+3]):
                slope = _c1_offset_slope(float(x))[1]
                tangent = np.array((1., 0., slope))
                tangent /= np.linalg.norm(tangent)
                transverse = np.array((0., 1., 0.))
                frames[local] = np.column_stack((
                    tangent, transverse, np.cross(tangent, transverse)))
        reference = CenteredCurvedBeam3ReferenceGeometry(coords, frames)
        native = NativeGeneralizedStaticElement(element+1, ids, reference,
                                                section, order=4)
        model.add_element(element+1, native)
        model.materials[native.material_name] = native.section
    model.add_boundary_condition(BoundaryCondition(
        'root', [1], {dof: 0. for dof in ('ux','uy','uz','rx','ry','rz')}))
    no_load = DistributedPattern(LinePattern(()), ())
    context = Context(model, Program((1.,), no_load))
    if assemble:
        residual, tangent = _sparse_native_assembly(context)
        oracle_r, oracle_k, _, _, _ = context.assemble(
            context.initial.mechanical, 0., context.initial.origins)
        parity = max(float(np.max(np.abs(oracle_r-residual))),
                     float(np.max(np.abs(oracle_k-tangent.toarray()))))
        if parity > 1e-9:
            raise AssertionError(f'native local sparse/dense parity failed: {parity}')
    else:
        residual, tangent, parity = None, None, None
    physical = np.array([i for i in range(context.nodal_count)
                         if i not in context.fixed], dtype=int)
    internal = np.arange(context.nodal_count, context.count)
    return Fixture(n_half_cells, model, context, residual, tangent,
                   physical, internal, parity, abscissae)
