"""Development-only two-body state/work bridge; no contact law or solve.

Full coordinates concatenate sphere then panel. Reduced coordinates retain
every unconstrained sphere DOF and the panel's existing rotating constraints.
The zero operator passed to the constraint builder extracts its kinematic map;
it is never used as a structural stiffness or solved.
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.corotational import build_nonlinear_rotating_mpc_map
from anyfesolver.nonlinear_restart import model_configuration_fingerprint


def _vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("expected finite one-dimensional authoritative DOF vector")
    return result.copy()


def _finite_history(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_history(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_history(item)
    elif isinstance(value, (np.ndarray, float, int, np.number)):
        if not np.all(np.isfinite(value)):
            raise ValueError("nonfinite element history")
    elif value is not None and not isinstance(value, (str, bool)):
        raise ValueError("unsupported element history type")


@dataclass
class BridgeState:
    """Owned disposable state, not a restart checkpoint or accepted solve."""
    owner: object
    displacement: np.ndarray
    sphere_history: dict
    panel_history: dict


class _Body:
    def __init__(self, model, label):
        self.model, self.label = model, label
        self.size = int(model.mesh.dof_manager.total_dofs)
        self.signature = self._signature()
        _, _, self.linear, self.offset, self.independent, self.info = (
            build_constraint_transformation(
                sparse.csr_matrix((self.size, self.size)), np.zeros(self.size), model))
        self.rotating = build_nonlinear_rotating_mpc_map(model, self.linear)
        self.check()

    def _signature(self):
        return (model_configuration_fingerprint(self.model),
                int(self.model.mesh.dof_manager.total_dofs),
                tuple((int(nid), tuple(node.dofs))
                      for nid, node in sorted(self.model.mesh.nodes.items())))

    def check(self):
        if self._signature() != self.signature:
            raise ValueError("stale model or authoritative DOF identity")

    def expand(self, reduced):
        full = np.asarray(self.linear @ reduced + self.offset).reshape(-1)
        return full if self.rotating is None else self.rotating.apply_kinematics(full)

    def coordinates(self, full):
        linear = full if self.rotating is None else self.rotating.to_linear_displacement(full)
        return linear[self.independent].copy()

    def jacobian(self, full):
        return self.linear.copy() if self.rotating is None else self.rotating.jacobian(full)

    def pullback(self, full, force):
        if self.rotating is None:
            return np.asarray(self.linear.T @ force).reshape(-1)
        return self.rotating.reduce_force(full, force)


class TwoBodyStateBridge:
    """Bind existing models and isolate trials from a copied accepted preload.

    No state can be committed here: successful equilibrium and acceptance are
    outside this component. Leaving a trial, including via cancellation or an
    exception, simply discards it. Callers receive copies of accepted history.
    """
    def __init__(self, sphere, panel, sphere_displacement, panel_displacement,
                 *, sphere_history=None, panel_history=None):
        if sphere is panel:
            raise ValueError("distinct body models required")
        self._bodies = (_Body(sphere, "sphere"), _Body(panel, "panel"))
        self._owner = object()
        self.sizes = tuple(b.size for b in self._bodies)
        self.reduced_sizes = tuple(b.linear.shape[1] for b in self._bodies)
        self.total_dofs = sum(self.sizes)
        self.reduced_dofs = sum(self.reduced_sizes)
        self._accepted = BridgeState(
            self._owner, self.pack(sphere, sphere_displacement, panel, panel_displacement),
            copy.deepcopy({} if sphere_history is None else sphere_history),
            copy.deepcopy({} if panel_history is None else panel_history))
        self._check_state(self._accepted)
        # Existing s8 postcheck tolerance, in the supplied displacement units.
        recovered = self.expand(self.reduced_coordinates(self._accepted))
        if np.max(np.abs(recovered - self._accepted.displacement), initial=0.) > 1e-12:
            raise ValueError("accepted displacement violates existing constraints")

    def _check_models(self):
        for body in self._bodies:
            body.check()

    def _check_state(self, state):
        self._check_models()
        if not isinstance(state, BridgeState) or state.owner is not self._owner:
            raise ValueError("foreign bridge state")
        _vector(state.displacement, self.total_dofs)
        for history in (state.sphere_history, state.panel_history):
            if not isinstance(history, dict):
                raise ValueError("element history must be a mapping")
            _finite_history(history)

    def pack(self, sphere, sphere_vector, panel, panel_vector):
        self._check_models()
        if sphere is not self._bodies[0].model or panel is not self._bodies[1].model:
            raise ValueError("wrong body identity or order")
        return np.concatenate((_vector(sphere_vector, self.sizes[0]),
                               _vector(panel_vector, self.sizes[1])))

    def unpack(self, full):
        self._check_models()
        vector = _vector(full, self.total_dofs)
        return vector[:self.sizes[0]].copy(), vector[self.sizes[0]:].copy()

    def accepted(self):
        self._check_state(self._accepted)
        return BridgeState(self._owner, self._accepted.displacement.copy(),
                           copy.deepcopy(self._accepted.sphere_history),
                           copy.deepcopy(self._accepted.panel_history))

    def reduced_coordinates(self, state):
        self._check_state(state)
        return np.concatenate([b.coordinates(u) for b, u in
                               zip(self._bodies, self.unpack(state.displacement))])

    def expand(self, reduced):
        self._check_models()
        q = _vector(reduced, self.reduced_dofs)
        parts = (q[:self.reduced_sizes[0]], q[self.reduced_sizes[0]:])
        full = np.concatenate([b.expand(p) for b, p in zip(self._bodies, parts)])
        return _vector(full, self.total_dofs)

    def jacobian(self, state):
        self._check_state(state)
        result = sparse.block_diag([b.jacobian(u) for b, u in
                                    zip(self._bodies, self.unpack(state.displacement))],
                                   format="csr")
        if not np.all(np.isfinite(result.data)):
            raise ValueError("nonfinite constraint Jacobian")
        return result

    def pullback(self, state, full_force):
        self._check_state(state)
        result = np.concatenate([b.pullback(u, f) for b, u, f in zip(
            self._bodies, self.unpack(state.displacement), self.unpack(full_force))])
        return _vector(result, self.reduced_dofs)

    @contextmanager
    def trial(self, reduced_increment=None):
        state = self.accepted()
        if reduced_increment is not None:
            state.displacement = self.expand(self.reduced_coordinates(state) +
                                            _vector(reduced_increment, self.reduced_dofs))
        try:
            yield state
        finally:
            # There is deliberately no commit path, even after normal return.
            self._check_models()

    def inventory(self):
        self._check_models()
        rows = []
        offset = 0
        for body in self._bodies:
            manager = body.model.mesh.dof_manager
            for dof in range(body.size):
                node, local, _ = manager.get_dof_info(dof)
                rows.append((body.label, int(node), int(local), dof, offset + dof))
            offset += body.size
        return tuple(rows)
