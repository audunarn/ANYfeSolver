"""Declared flat-Q4 supported variations and their dual work map.

This private development map compares a *shared field family*, not complete
native/external coordinate spaces. All coordinates outside the selected shell
are present and zero, including every signed native retained role. The physical
field is the nodal Q1 displacement/linearized rotation interpolant. Constitutive
and internal strain reconstructions remain the responsibility of each operator.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportedShellWorkMap:
    """q = evaluation @ a, with dimensionless virtual amplitudes a.

    Translation rows carry m; rotation rows carry rad. Consequently a physical
    dual covector pulls back to J per unit amplitude. This class does not turn
    mixed signed work into a norm or assume an invertible transfer.
    """

    evaluation: np.ndarray
    names: tuple[str, ...]
    shell_slots: tuple[int, ...]
    zero_slots: tuple[int, ...]

    def __post_init__(self):
        p = np.asarray(self.evaluation, dtype=float)
        if (p.ndim != 2 or p.shape[1] != len(self.names) or
                not np.all(np.isfinite(p)) or len(set(self.names)) != len(self.names)):
            raise ValueError("finite supported-field evaluation and unique names required")
        if (any(type(i) is not int or not 0 <= i < len(p) for i in self.shell_slots + self.zero_slots) or
                len(set(self.shell_slots)) != len(self.shell_slots) or
                len(set(self.zero_slots)) != len(self.zero_slots) or
                np.any(p[list(self.zero_slots)] != 0.)):
            raise ValueError("supported-field zero slots must vanish exactly")
        outside = sorted(set(range(len(p))) - set(self.shell_slots))
        if np.any(p[outside] != 0.):
            raise ValueError("shell-only variation cannot populate beam or retained roles")
        # Bytes-backed storage prevents mutation after the comparison is bound.
        owned = np.frombuffer(np.ascontiguousarray(p).tobytes(), dtype=float).reshape(p.shape)
        object.__setattr__(self, "evaluation", owned)

    def evaluate(self, amplitudes):
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (len(self.names),) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("one finite dimensionless amplitude per declared field required")
        return self.evaluation @ amplitudes

    def transpose_work(self, dual):
        dual = np.asarray(dual, dtype=float)
        if dual.shape != (len(self.evaluation),) or not np.all(np.isfinite(dual)):
            raise ValueError("full finite physical dual covector required")
        return self.evaluation.T @ dual

    def body_work(self, operator):
        """Return the signed bilinear work matrix in J, preserving raw operator."""
        if operator.shape != (len(self.evaluation), len(self.evaluation)):
            raise ValueError("full body operator dimensions do not match evaluation")
        action = operator @ self.evaluation
        if not np.all(np.isfinite(action)):
            raise ValueError("nonfinite supported-field body action")
        return self.evaluation.T @ action

    def require_constraint_kernel(self, constraints):
        """Require exact zero trace action, stronger than work at one density.

        The constructed fields have identically zero attachment rows, so no
        mixed-unit tolerance or numerical cancellation is needed here.
        """
        if constraints.shape[1] != len(self.evaluation):
            raise ValueError("full constraint columns do not match evaluation")
        if hasattr(constraints, "tocoo"):
            entries = constraints.tocoo()
            columns = np.unique(entries.col[entries.data != 0.])
            finite = np.all(np.isfinite(entries.data))
        else:
            values = np.asarray(constraints, dtype=float)
            columns = np.flatnonzero(np.any(values != 0., axis=0))
            finite = np.all(np.isfinite(values))
        if not finite or np.any(self.evaluation[columns] != 0.):
            raise ValueError("supported variation is outside the structural exact trace kernel")
        gap = constraints @ self.evaluation
        if not np.all(np.isfinite(gap)) or np.any(gap != 0.):
            raise ValueError("supported variation is outside the exact trace kernel")
        return gap


def flat_q4_supported_shell_map(*, coordinates, node_slots, full_count,
                                fixed_slots, trace_nodes, cells,
                                translation_m=1., rotation_rad=1.):
    """Build six interior component fields on a declared flat rectangle.

    phi=16*xi**2*(1-xi)**2*eta**2*(1-eta)**2 is merely a nodal generator;
    the represented field is its Q1 interpolant. Boundary values are explicitly
    zero. Every trace node must lie on the rectangle boundary. This intentionally
    refuses curved, non-Q4, interior-line and incomplete supported maps.
    """
    xyz = np.asarray(coordinates, dtype=float)
    slots = np.asarray(node_slots)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 9 or
            not np.all(np.isfinite(xyz)) or slots.shape != (len(xyz), 6) or
            not np.issubdtype(slots.dtype, np.integer) or
            type(full_count) is not int or full_count <= 0 or
            len(np.unique(slots)) != slots.size or np.any(slots < 0) or np.any(slots >= full_count)):
        raise ValueError("finite shell nodes with six unique full slots required")
    if (not np.isfinite(translation_m) or translation_m <= 0 or
            not np.isfinite(rotation_rad) or rotation_rad <= 0):
        raise ValueError("positive finite m/rad virtual-field scales required")
    if np.any(xyz[:, 2] != xyz[0, 2]):
        raise ValueError("only the declared flat Cartesian Q4 comparison is admitted")
    xs, ys = np.unique(xyz[:, 0]), np.unique(xyz[:, 1])
    if len(xs) < 3 or len(ys) < 3 or len(xs)*len(ys) != len(xyz):
        raise ValueError("complete rectangular Q4 grid with interior nodes required")
    lookup = {(float(x), float(y)): i for i, (x, y, _) in enumerate(xyz)}
    if len(lookup) != len(xyz):
        raise ValueError("duplicate physical shell node")
    expected = {tuple(lookup[(float(x), float(y))] for x, y in
                      ((xs[i], ys[j]), (xs[i+1], ys[j]),
                       (xs[i+1], ys[j+1]), (xs[i], ys[j+1])))
                for j in range(len(ys)-1) for i in range(len(xs)-1)}
    actual = tuple(tuple(int(i) for i in cell) for cell in cells)
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("exact ordered Q4 cell correspondence required")
    boundary = {i for i, (x, y, _) in enumerate(xyz)
                if x in (xs[0], xs[-1]) or y in (ys[0], ys[-1])}
    trace_nodes = tuple(trace_nodes)
    if (not trace_nodes or any(type(i) is not int or i not in boundary for i in trace_nodes) or
            len(set(trace_nodes)) != len(trace_nodes)):
        raise ValueError("distinct boundary attachment nodes required")
    p = np.zeros((full_count, 6))
    length, width = xs[-1]-xs[0], ys[-1]-ys[0]
    for i, (x, y, _) in enumerate(xyz):
        if i in boundary:
            continue
        xi, eta = (x-xs[0])/length, (y-ys[0])/width
        phi = 16.*xi**2*(1.-xi)**2*eta**2*(1.-eta)**2
        for component in range(6):
            p[slots[i, component], component] = phi*(translation_m if component < 3 else rotation_rad)
    zero_slots = tuple(sorted(set(int(v) for v in fixed_slots) |
                              {int(v) for i in boundary for v in slots[i]}))
    return SupportedShellWorkMap(p,
        ("membrane_ux", "membrane_uy", "transverse_w", "rotation_rx", "rotation_ry", "normal_spin_rz"),
        tuple(int(v) for v in slots.ravel()), zero_slots)
