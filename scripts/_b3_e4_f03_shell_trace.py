"""Planar physical shell and exact reference trace for the F03-K0 diagnostic.

The trace uses only nodal in-plane translations.  It is the linearization of
the proper, centered four-point Procrustes fit at the flat reference state;
it does not introduce a shell drilling coordinate or an artificial spring.
This helper is preparatory and is not a registered execution or product API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material


_HALF_WINDOW = 0.2
_HALF_RAIL = 0.23
_SAMPLE_OFFSETS = ((-_HALF_WINDOW, -_HALF_RAIL),
                   (_HALF_WINDOW, -_HALF_RAIL),
                   (_HALF_WINDOW, _HALF_RAIL),
                   (-_HALF_WINDOW, _HALF_RAIL))
_YAW_DENOMINATOR = 4.0 * (_HALF_WINDOW**2 + _HALF_RAIL**2)


@dataclass
class ShellPanel:
    """Full nodal in-plane stiffness and supported map of one shell panel.

    ``trace(s)`` has columns in the same global [ux0,uy0,ux1,uy1,...]
    order as ``stiffness``.  Restrict the columns with ``free`` when forming
    a supported operator.  ``breakpoints`` includes the attachment ends.
    """

    n_s: int
    topology: str
    stiffness: sparse.csr_matrix
    free: np.ndarray
    node_count: int
    element_count: int
    thickness: float = .07
    n_y: int = 2

    @property
    def dof_count(self) -> int:
        return 2 * self.node_count

    @property
    def free_count(self) -> int:
        return len(self.free)

    @property
    def x_interval_count(self) -> int:
        return 2 * self.n_s

    def _shape(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Physical Q4 bilinear or diagonally split S3 barycentric host."""
        nx = self.x_interval_count
        dx = 1.0 / self.n_s
        dy = 1.0 / self.n_y
        # At exact edges choose the cell to the left/below.  Both hosts give
        # identical continuous shape traces there; the choice is deterministic.
        i = int(np.clip(np.ceil((x + .5) / dx) - 1, 0, nx - 1))
        j = int(np.clip(np.ceil((y + .5) / dy) - 1, 0, self.n_y - 1))
        xi = (x - (-.5 + i * dx)) / dx
        eta = (y - (-.5 + j * dy)) / dy
        if not (-1e-12 <= xi <= 1 + 1e-12 and
                -1e-12 <= eta <= 1 + 1e-12):
            raise ValueError("sample is outside its shell cell")
        a = j * (nx + 1) + i
        b, d = a + 1, a + nx + 1
        c = d + 1
        if self.topology == "Q4":
            return (np.array((a, b, c, d)),
                    np.array(((1-xi)*(1-eta), xi*(1-eta), xi*eta,
                              (1-xi)*eta)))
        if eta <= xi:
            return (np.array((a, b, c)),
                    np.array((1-xi, xi-eta, eta)))
        return (np.array((a, c, d)),
                np.array((1-eta, xi, eta-xi)))

    def sample_weights(self, s: float) -> np.ndarray:
        """Four exact host rows against global physical shell nodes."""
        if not 0.0 <= s <= 1.0:
            raise ValueError("attachment coordinate s must be in [0, 1]")
        weights = np.zeros((4, self.node_count))
        for k, (ox, y) in enumerate(_SAMPLE_OFFSETS):
            indices, values = self._shape(s + ox, y)
            weights[k, indices] = values
        return weights

    def sample_host_sizes(self, s: float) -> tuple[int, int, int, int]:
        """Element host node counts, including possible edge zero weights."""
        return tuple(len(self._shape(s + ox, y)[0])
                     for ox, y in _SAMPLE_OFFSETS)

    def trace(self, s: float) -> np.ndarray:
        """Analytic reference [centroid-x, centroid-y, proper-fit yaw] map.

        For centered sample coordinates r_k=(ox_k,y_k), the infinitesimal
        Procrustes yaw is Σ(r_x δu_y-r_y δu_x)/Σ|r|².  This automatically
        removes centroid motion, since Σr_k=0.  It agrees with the derivative
        of the orientation-preserving polar factor (det R=+1) at reference.
        """
        weights = self.sample_weights(s)
        out = np.zeros((3, self.dof_count))
        out[0, 0::2] = weights.mean(axis=0)
        out[1, 1::2] = weights.mean(axis=0)
        for k, (ox, y) in enumerate(_SAMPLE_OFFSETS):
            out[2, 0::2] -= y * weights[k] / _YAW_DENOMINATOR
            out[2, 1::2] += ox * weights[k] / _YAW_DENOMINATOR
        return out

    def breakpoints(self) -> np.ndarray:
        """All attachment knots at sample-host edges and S3 diagonals."""
        dx = 1.0 / self.n_s
        candidates = [0.0, 1.0]
        for ox, y in _SAMPLE_OFFSETS:
            eta = (y + .5) * self.n_y
            eta -= np.floor(eta)
            for i in range(self.x_interval_count + 1):
                edge = -.5 + i * dx
                candidates.append(edge - ox)
                if self.topology == "S3-V2D" and i < self.x_interval_count:
                    candidates.append(edge + eta * dx - ox)
        # Round only the knot representation: the physical sample/shape
        # arithmetic remains full precision.  Prevent near-duplicate slivers.
        return np.unique(np.round([max(0., min(1., v)) for v in candidates
                                   if -1e-12 <= v <= 1 + 1e-12], 14))

    def diagnostics(self) -> dict:
        """Mesh and actual reference-sample host counts for audit output."""
        return dict(topology=self.topology, n_s=self.n_s,
                    x_intervals=self.x_interval_count, y_intervals=self.n_y,
                    nodes=self.node_count, elements=self.element_count,
                    dofs=self.dof_count, free_dofs=self.free_count,
                    supported_dofs=self.dof_count-self.free_count,
                    sample_hosts=self.sample_host_sizes(.375),
                    shell_knots=len(self.breakpoints()))


def make_shell(n_s: int, topology: str, *, thickness: float = .07,
               n_y: int = 2) -> ShellPanel:
    """Assemble actual E4-PL physical in-plane blocks on a bounded grid.

    x spans [-.5, 1.5] in steps of 1/n_s; y spans [-.5, .5] in
    n_y rows. Only both x-end edges have ux and uy fixed; the
    long top and bottom rails remain free to deform and transfer
    the transverse physical torque arm. Nodal out-of-plane and
    rotational coordinates never enter the assembled membrane matrix.
    """
    if isinstance(n_s, bool) or not isinstance(n_s, (int, np.integer)) or n_s < 1:
        raise ValueError("n_s must be a positive integer")
    if isinstance(n_y, bool) or not isinstance(n_y, int) or n_y not in (2, 4, 8):
        raise ValueError("n_y must be 2, 4, or 8 in this bounded fixture")
    if topology not in ("Q4", "S3-V2D"):
        raise ValueError("topology must be Q4 or S3-V2D")
    if (isinstance(thickness, bool) or
            not isinstance(thickness, (int, float, np.integer, np.floating)) or
            not np.isfinite(thickness) or thickness <= 0.):
        raise ValueError("finite positive shell thickness required")
    nx = 2 * int(n_s)
    x_coords = np.linspace(-.5, 1.5, nx + 1)
    y_coords = np.linspace(-.5, .5, n_y + 1)
    mesh = FEMesh()
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            mesh.add_node(1 + j * (nx+1) + i, float(x), float(y), 0.)
    material = Material("steel", 210e9, .3, density=7850.)
    rows, cols, values = [], [], []
    element_count = 0
    for j in range(n_y):
        for i in range(nx):
            a = 1 + j * (nx+1) + i
            b, d = a+1, a+nx+1
            c = d+1
            patches = ((a, b, c, d),) if topology == "Q4" else (
                (a, b, c), (a, c, d))
            for nodes in patches:
                element_count += 1
                shell = create_shell_element(
                    element_count, list(nodes), "steel",
                    formulation="e4-pl" if topology == "Q4" else "e4-pl-s3-v2d",
                    thickness=float(thickness), reference_normal=(0., 0., 1.))
                physical = np.asarray(
                    shell.compute_stiffness_components(mesh, material)["physical"])
                local = [6*k+axis for k in range(len(nodes)) for axis in (0, 1)]
                block = physical[np.ix_(local, local)]
                dofs = [2*(node-1)+axis for node in nodes for axis in (0, 1)]
                for local_row, global_row in enumerate(dofs):
                    for local_col, global_col in enumerate(dofs):
                        rows.append(global_row)
                        cols.append(global_col)
                        values.append(block[local_row, local_col])
    node_count = (n_y+1) * (nx+1)
    stiffness = sparse.coo_matrix(
        (values, (rows, cols)), shape=(2*node_count, 2*node_count)).tocsr()
    free = np.array([2*(j*(nx+1)+i)+axis
                     for j in range(n_y+1) for i in range(1, nx)
                     for axis in (0, 1)], dtype=int)
    return ShellPanel(int(n_s), topology, stiffness, free, node_count,
                      element_count, float(thickness), n_y)
