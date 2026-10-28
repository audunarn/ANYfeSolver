"""
Boundary Conditions and Load Cases

This module provides classes for defining boundary conditions and loads
for the FE model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABCMeta
from contextlib import ExitStack
import io
import codecs
from encodings import ascii as _ASCII_CODEC, cp1252 as _CP1252_CODEC
from encodings import latin_1 as _LATIN1_CODEC, utf_8 as _UTF8_CODEC
from gc import get_referents as _GET_REFERENTS
import linecache
import re
import sys
import types
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy._core import _ufunc_config as _NUMPY_ERROR_MODULE

from .matrix_assembly import (
    _CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE as _CAPTURE_QUALIFIED_LOAD_RUNTIME_LEASE,
)
from .e4_pl_element import (
    QualifiedE4PLShellElement as _QualifiedE4PLShellElement,
    _q4_trusted_operation_scope as _Q4_TRUSTED_OPERATION_SCOPE,
)
from .e4_pl_s3_element import (
    QualifiedE4PLS3ShellElement as _QualifiedE4PLS3ShellElement,
    _s3_trusted_operation_scope as _S3_TRUSTED_OPERATION_SCOPE,
)

from ._qualified_authority_epoch import _AuthorityEpochManager
from . import elements as _element_module
from .elements import (
    BeamElement as _BeamElement,
    CoupledBeamShellElement as _CoupledBeamShellElement,
    Element as _Element,
    ShellElement as _ShellElement,
)
from anymaterial import IsotropicMaterial as _IsotropicMaterial

from .fe_core import (
    DOFManager as _DOFManager, FEModel as _FEModel, FEMesh as _FEMesh, Node as _Node,
    _QualifiedStateMapping,
)

if TYPE_CHECKING:
    from .fe_core import DOFManager, FEMesh, Material, Node


_SMALL = 1.0e-12


class _MeshLeaseModel:
    """Closure-private adapter for direct mesh-level load operations."""

    __slots__ = ("mesh",)

    def __init__(self, mesh: "FEMesh") -> None:
        self.mesh = mesh


def _run_with_qualified_load_runtime_lease(
    mesh: "FEMesh",
    *,
    context: str,
    operation: Callable[[Callable[..., None]], Any],
) -> Any:
    proxy = _MeshLeaseModel(mesh)
    lease = _CAPTURE_QUALIFIED_LOAD_RUNTIME_LEASE(
        proxy,
        context=f"{context} preflight",
    )

    def require(*, stage: str) -> None:
        lease(proxy, context=f"{context} {stage}")

    try:
        qualified_elements = tuple(mesh.elements.values())
        q4_elements = tuple(
            element
            for element in qualified_elements
            if type(element) is _QualifiedE4PLShellElement
        )
        s3_elements = tuple(
            element
            for element in qualified_elements
            if type(element) is _QualifiedE4PLS3ShellElement
        )
        with ExitStack() as stack:
            stack.enter_context(_Q4_TRUSTED_OPERATION_SCOPE(q4_elements))
            stack.enter_context(_S3_TRUSTED_OPERATION_SCOPE(s3_elements))
            result = operation(require)
    except BaseException:
        require(stage="exceptional output")
        raise
    require(stage="output")
    return result


class _GravityFallbackMaterial:
    """Minimal material used when loads are assembled without model context."""

    density = 7850.0
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3

    @property
    def shear_modulus(self) -> float:
        return self.elastic_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class BoundaryCondition:
    """
    Base class for boundary conditions.

    Boundary conditions constrain specific DOFs of nodes.
    """

    name: str
    node_ids: List[int]
    dof_constraints: Dict[str, float]

    def __post_init__(self):
        dof_names = ["ux", "uy", "uz", "rx", "ry", "rz"]
        self._dof_indices = {}
        for dof_name, value in self.dof_constraints.items():
            if dof_name in dof_names:
                self._dof_indices[dof_name] = dof_names.index(dof_name)

    def apply(self, dof_manager: "DOFManager"):
        """Apply this boundary condition to the DOF manager."""
        for node_id in self.node_ids:
            node_dofs = dof_manager.get_node_dofs(node_id)
            if not node_dofs:
                continue
            for dof_name, value in self.dof_constraints.items():
                if dof_name in self._dof_indices:
                    local_idx = self._dof_indices[dof_name]
                    global_dof = node_dofs[local_idx]
                    dof_manager.constrain_dof(global_dof)

    def get_constrained_dofs(self, dof_manager: "DOFManager") -> List[Tuple[int, float]]:
        """Get list of (global_dof, prescribed_value) pairs."""
        constrained = []
        for node_id in self.node_ids:
            node_dofs = dof_manager.get_node_dofs(node_id)
            if not node_dofs:
                continue
            for dof_name, value in self.dof_constraints.items():
                if dof_name in self._dof_indices:
                    local_idx = self._dof_indices[dof_name]
                    global_dof = node_dofs[local_idx]
                    constrained.append((global_dof, value))
        return constrained


@dataclass
class FixedSupport(BoundaryCondition):
    """Fully fixed support: all DOFs constrained to zero."""

    def __init__(self, name: str, node_ids: List[int]):
        dof_constraints = {dof: 0.0 for dof in ["ux", "uy", "uz", "rx", "ry", "rz"]}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class PinnedSupport(BoundaryCondition):
    """Pinned support: translational DOFs fixed, rotations free."""

    def __init__(self, name: str, node_ids: List[int]):
        dof_constraints = {dof: 0.0 for dof in ["ux", "uy", "uz"]}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class RollerSupport(BoundaryCondition):
    """Roller support constraining selected translational DOFs."""

    def __init__(self, name: str, node_ids: List[int], constrained_directions: Optional[List[str]] = None):
        if constrained_directions is None:
            constrained_directions = ["uy", "uz"]
        dof_constraints = {dof: 0.0 for dof in constrained_directions}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class SymmetryBC(BoundaryCondition):
    """Symmetry boundary condition in a global coordinate plane."""

    def __init__(self, name: str, node_ids: List[int], symmetry_plane: str = "xy"):
        if symmetry_plane == "xy":
            dof_constraints = {"uz": 0.0, "rx": 0.0, "ry": 0.0}
        elif symmetry_plane == "xz":
            dof_constraints = {"uy": 0.0, "rx": 0.0, "rz": 0.0}
        elif symmetry_plane == "yz":
            dof_constraints = {"ux": 0.0, "ry": 0.0, "rz": 0.0}
        else:
            dof_constraints = {}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class LoadCase:
    """
    Load case for the FE model.

    Contains nodal loads, element loads, pressure loads and optional gravity.
    Pressure loads on shell elements are assembled as consistent nodal loads by
    Gauss integration over the element surface, instead of equal area shares.
    Pressure is a reference-configuration dead load by default.
    ``follower_pressure=True`` switches every pressure in the load case to the
    current-area formulation used by nonlinear static and arc-length analysis.
    """

    name: str
    nodal_loads: Dict[int, np.ndarray] = field(default_factory=dict)
    element_loads: Dict[int, np.ndarray] = field(default_factory=dict)
    pressure_loads: Dict[int, Any] = field(default_factory=dict)
    gravity: Optional[np.ndarray] = None
    added_node_masses: Dict[int, float] = field(default_factory=dict)
    follower_pressure: bool = False

    def add_nodal_load(
        self,
        node_id: int,
        load_vector: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        moments: Optional[np.ndarray] = None,
    ):
        """
        Add a nodal load.

        Args:
            node_id: Node ID to apply load to.
            load_vector: [Fx, Fy, Fz] or [Fx, Fy, Fz, Mx, My, Mz].
            forces: Alternative force vector [Fx, Fy, Fz].
            moments: Optional moment vector [Mx, My, Mz].
        """
        if load_vector is not None:
            load_vector = np.asarray(load_vector, dtype=float)
            if len(load_vector) == 6:
                load = load_vector.copy()
            else:
                if moments is None:
                    moments = np.zeros(3)
                load = np.concatenate([load_vector[:3], np.asarray(moments, dtype=float)[:3]])
        elif forces is not None:
            if moments is None:
                moments = np.zeros(3)
            load = np.concatenate([np.asarray(forces, dtype=float)[:3], np.asarray(moments, dtype=float)[:3]])
        elif moments is not None:
            load = np.concatenate([np.zeros(3), np.asarray(moments, dtype=float)[:3]])
        else:
            load = np.zeros(6)

        if node_id in self.nodal_loads:
            self.nodal_loads[node_id] += load
        else:
            self.nodal_loads[node_id] = load

    def add_pressure_load(self, element_id: int, pressure: Any):
        """
        Add a pressure load to a shell element.

        Positive pressure follows the element normal as defined by the element
        node ordering and natural-coordinate surface Jacobian.  A scalar is a
        uniform pressure.  Exactly three finite nodal values may be supplied
        for an affine field; that field is accepted only by an element whose
        formulation explicitly owns affine pressure mechanics.

        Pressure is a dead load by default.  Set ``follower_pressure=True`` on
        the load case to integrate it over the current nodal interpolation
        surface during a
        nonlinear solve.  The flag is load-case-wide so one proportional load
        pattern cannot accidentally mix reference- and current-configuration
        pressure semantics.
        """
        try:
            scalar = float(pressure)
        except (TypeError, ValueError):
            try:
                values = np.asarray(pressure, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "pressure must be a scalar or exactly three finite nodal values"
                ) from exc
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "pressure must be a scalar or exactly three finite nodal values"
                )
            # Own the field at registration so later caller mutation cannot
            # change a load case between authority checks and assembly.
            self.pressure_loads[element_id] = tuple(float(value) for value in values)
        else:
            self.pressure_loads[element_id] = scalar

    @staticmethod
    def _is_strict_flat_s3_v2(element: Any) -> bool:
        """Return true only for the exact opt-in Stage-1 V2 production class."""

        # Keep the dependency local: the V2 module derives from ShellElement,
        # while boundary/load definitions are imported by the public package.
        from .e4_pl_s3_v2_element import (
            StrictFlatLinearE4PLS3V2ShellElement,
        )

        return type(element) is StrictFlatLinearE4PLS3V2ShellElement

    @classmethod
    def _strict_flat_s3_v2_dead_pressure_load(
        cls,
        element: Any,
        mesh: "FEMesh",
        pressure: Any,
    ) -> Optional[np.ndarray]:
        """Dispatch the exact V2 dead-pressure kernel without legacy fallback."""

        if not cls._is_strict_flat_s3_v2(element):
            return None
        return np.asarray(
            element.compute_dead_transverse_pressure_load(mesh, pressure),
            dtype=float,
        )

    @classmethod
    def _reject_strict_flat_s3_v2_follower_pressure(cls, element: Any) -> None:
        if not cls._is_strict_flat_s3_v2(element):
            return
        from .e4_pl_s3_v2_element import StrictFlatLinearCapabilityError

        raise StrictFlatLinearCapabilityError(
            "strict-flat S3 V2 capability 'follower_pressure' is outside the "
            "authorized Stage-1 flat-linear surface"
        )

    def set_gravity(self, gx: float = 0.0, gy: float = 0.0, gz: float = -9.81):
        """Set gravity acceleration."""
        self.gravity = np.array([gx, gy, gz], dtype=float)

    def set_acceleration(self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0):
        """Set a body-load acceleration field in x/y/z.

        Produces the consistent inertial load ``M a`` over the structural mass
        (element mass matrices) plus ``m_i a`` for any added nodal masses.  This
        is the same mechanism as :meth:`set_gravity`; use it to describe design
        accelerations (e.g. ship motions) in an arbitrary direction.
        """
        self.gravity = np.array([ax, ay, az], dtype=float)

    def add_node_mass(self, node_id: int, mass: float):
        """Add a lumped translational mass at a node.

        The added mass contributes an inertial load ``mass * acceleration`` at
        the node's translational DOFs whenever an acceleration/gravity field is
        set.  Use the frontend edge/ring helpers to distribute a total mass
        along a plate edge or a cylinder top/bottom ring.
        """
        mass = float(mass)
        if mass == 0.0:
            return
        self.added_node_masses[int(node_id)] = self.added_node_masses.get(int(node_id), 0.0) + mass

    def add_distributed_edge_mass(self, node_ids: Sequence[int], total_mass: float):
        """Distribute ``total_mass`` equally over the given nodes."""
        node_ids = [int(node_id) for node_id in node_ids]
        if not node_ids or float(total_mass) == 0.0:
            return
        share = float(total_mass) / float(len(node_ids))
        for node_id in node_ids:
            self.add_node_mass(node_id, share)

    @staticmethod
    def _surface_jacobian_and_normal(coords: np.ndarray, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute surface Jacobian magnitude and unit normal from shape derivatives.
        """
        tangent_xi = coords.T @ dN_dxi
        tangent_eta = coords.T @ dN_deta
        normal_raw = np.array(
            [
                tangent_xi[1] * tangent_eta[2] - tangent_xi[2] * tangent_eta[1],
                tangent_xi[2] * tangent_eta[0] - tangent_xi[0] * tangent_eta[2],
                tangent_xi[0] * tangent_eta[1] - tangent_xi[1] * tangent_eta[0],
            ]
        )
        det_j = float(np.linalg.norm(normal_raw))
        if det_j < _SMALL:
            return 0.0, np.array([0.0, 0.0, 1.0])
        return det_j, normal_raw / det_j

    @staticmethod
    def _fallback_lumped_pressure_load(
        element,
        mesh: "FEMesh",
        pressure: float,
        coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fallback for unsupported element topologies.

        This keeps old behaviour available for non-shell or future custom elements,
        but all 4/8-node shell elements should use the consistent path.
        """
        if coords is None:
            coords = element.get_node_coordinates(mesh)
        coords = np.asarray(coords, dtype=float)
        num_nodes = len(element.node_ids)
        f_elem = np.zeros(num_nodes * 6)
        if num_nodes < 3:
            return f_elem

        if num_nodes in (4, 8):
            tri1_area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            tri2_area = 0.5 * np.linalg.norm(np.cross(coords[0] - coords[2], coords[3] - coords[2]))
            area = tri1_area + tri2_area
        else:
            area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))

        normal_raw = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        normal_norm = np.linalg.norm(normal_raw)
        normal = normal_raw / normal_norm if normal_norm > _SMALL else np.array([0.0, 0.0, 1.0])
        nodal_force = pressure * area / max(num_nodes, 1) * normal
        for i in range(num_nodes):
            f_elem[i * 6:i * 6 + 3] += nodal_force
        return f_elem

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        """Return ``[vector]_x`` such that ``[vector]_x @ x = vector x x``."""
        x, y, z = np.asarray(vector, dtype=float).reshape(3)
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _current_element_coordinates(
        element,
        mesh: "FEMesh",
        displacements: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return nodal interpolation-surface coordinates after translations."""
        coords = np.asarray(element.get_node_coordinates(mesh), dtype=float).copy()
        if displacements is None:
            return coords
        u = np.asarray(displacements, dtype=float).reshape(-1)
        for local_index, node_id in enumerate(element.node_ids):
            node = mesh.get_node(int(node_id))
            if node is None:
                continue
            translational_dofs = np.asarray(node.dofs[:3], dtype=np.intp)
            if translational_dofs.size == 3 and int(np.max(translational_dofs)) < u.size:
                coords[local_index] += u[translational_dofs]
        return coords

    def _consistent_pressure_load(
        self,
        element,
        mesh: "FEMesh",
        pressure: float,
        coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Assemble a consistent element pressure vector.

        For a shell element with shape functions N_i, the translational nodal
        load is:

            f_i = integral_A N_i * p * n dA

        When ``coords`` are current nodal interpolation-surface coordinates this is a follower
        load.  No independent rotational pressure moments are introduced:
        pressure virtual work is conjugate to interpolation-surface translations.
        """
        if coords is None:
            native_dead = getattr(element, "compute_native_dead_pressure_load", None)
            if callable(native_dead):
                return np.asarray(native_dead(mesh, pressure), dtype=float)
        else:
            native_current = getattr(
                element, "compute_native_current_pressure_load", None
            )
            if callable(native_current):
                return np.asarray(
                    native_current(mesh, pressure, coords), dtype=float
                )

        v2_load = self._strict_flat_s3_v2_dead_pressure_load(
            element,
            mesh,
            pressure,
        )
        if v2_load is not None:
            return v2_load

        try:
            pressure = float(pressure)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "affine nodal pressure requires a formulation-specific dead-pressure kernel"
            ) from exc
        if not hasattr(element, "compute_shape_functions") or not hasattr(element, "gauss_points"):
            return self._fallback_lumped_pressure_load(element, mesh, pressure, coords)

        if coords is None:
            coords = element.get_node_coordinates(mesh)
        coords = np.asarray(coords, dtype=float)
        num_nodes = len(element.node_ids)
        f_elem = np.zeros(num_nodes * 6)
        gauss_points = getattr(element, "gauss_points")
        gauss_weights = getattr(element, "gauss_weights")
        orientation_provider = getattr(
            element, "sheet_area_orientation_sign", None
        )
        orientation_sign = (
            float(orientation_provider(mesh))
            if callable(orientation_provider)
            else 1.0
        )
        if orientation_sign not in (-1.0, 1.0):
            raise ValueError("shell sheet area orientation sign must be -1 or +1")

        for (xi, eta), weight in zip(gauss_points, gauss_weights):
            N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
            tangent_xi = coords.T @ dN_dxi
            tangent_eta = coords.T @ dN_deta
            area_vector = orientation_sign * np.cross(tangent_xi, tangent_eta)
            if float(np.linalg.norm(area_vector)) < _SMALL:
                continue
            for i in range(num_nodes):
                f_elem[i * 6:i * 6 + 3] += N[i] * pressure * area_vector * float(weight)
        return f_elem

    def _consistent_pressure_tangent(
        self,
        element,
        mesh: "FEMesh",
        pressure: float,
        coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the exact current-area follower-pressure load tangent.

        For ``f_i = integral N_i p (a_xi x a_eta) dxi deta``, the 3x3 block
        derivative with respect to node ``j`` is

        ``p N_i (-N_j,xi [a_eta]_x + N_j,eta [a_xi]_x)``.

        The matrix is generally nonsymmetric for an open pressure patch.  It is
        the derivative of the external force, so equilibrium Newton systems use
        ``K_internal - K_external``.
        """
        self._reject_strict_flat_s3_v2_follower_pressure(element)
        native_tangent = getattr(
            element, "compute_native_current_pressure_tangent", None
        )
        if callable(native_tangent):
            if coords is None:
                coords = self._current_element_coordinates(
                    element,
                    mesh,
                    np.zeros(mesh.dof_manager.total_dofs, dtype=float),
                )
            return np.asarray(
                native_tangent(mesh, pressure, coords), dtype=float
            )
        if not hasattr(element, "compute_shape_functions") or not hasattr(element, "gauss_points"):
            raise ValueError(
                f"Follower pressure requires a shell interpolation with shape functions; "
                f"element {getattr(element, 'element_id', '?')} is unsupported."
            )

        if coords is None:
            coords = element.get_node_coordinates(mesh)
        coords = np.asarray(coords, dtype=float)
        num_nodes = len(element.node_ids)
        tangent = np.zeros((num_nodes * 6, num_nodes * 6), dtype=float)
        gauss_points = getattr(element, "gauss_points")
        gauss_weights = getattr(element, "gauss_weights")
        orientation_provider = getattr(
            element, "sheet_area_orientation_sign", None
        )
        orientation_sign = (
            float(orientation_provider(mesh))
            if callable(orientation_provider)
            else 1.0
        )
        if orientation_sign not in (-1.0, 1.0):
            raise ValueError("shell sheet area orientation sign must be -1 or +1")

        for (xi, eta), weight in zip(gauss_points, gauss_weights):
            N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
            tangent_xi = coords.T @ dN_dxi
            tangent_eta = coords.T @ dN_deta
            if float(np.linalg.norm(np.cross(tangent_xi, tangent_eta))) < _SMALL:
                continue
            skew_xi = self._skew(tangent_xi)
            skew_eta = self._skew(tangent_eta)
            scale = orientation_sign * float(pressure) * float(weight)
            for i in range(num_nodes):
                row = slice(6 * i, 6 * i + 3)
                for j in range(num_nodes):
                    col = slice(6 * j, 6 * j + 3)
                    tangent[row, col] += (
                        scale
                        * float(N[i])
                        * (-float(dN_dxi[j]) * skew_eta + float(dN_deta[j]) * skew_xi)
                    )
        return tangent

    def _consistent_gravity_load(
        self,
        element,
        mesh: "FEMesh",
        material: "Material",
    ) -> np.ndarray:
        """
        Assemble element body force from the element mass matrix.

        With translational acceleration a, the consistent nodal load is M a.
        Rotational acceleration components are zero for ordinary gravity.
        """
        f_elem = np.zeros(len(element.node_ids) * 6)
        if self.gravity is None:
            return f_elem

        mass_matrix = element.compute_mass_matrix(mesh, material)
        acceleration = np.zeros_like(f_elem)
        for i in range(len(element.node_ids)):
            acceleration[i * 6:i * 6 + 3] = self.gravity
        return np.asarray(mass_matrix @ acceleration, dtype=float).reshape(-1)

    def _get_load_vector_under_lease(
        self,
        mesh: "FEMesh",
        dof_manager: "DOFManager",
        material_getter: Optional[Callable[[str], "Material"]] = None,
        displacements: Optional[np.ndarray] = None,
        element_activity: Optional[object] = None,
        *,
        qualified_runtime_guard: Callable[..., None],
    ) -> np.ndarray:
        """Assemble the global load vector.

        ``displacements`` affects only pressure loads when
        :attr:`follower_pressure` is true.  Dead pressure, nodal, element and
        gravity loads retain their reference-configuration semantics.
        """
        qualified_runtime_guard(stage="load-vector preflight")
        if any(type(e).__module__ == "anysolver._ge_beam3_native_fibre_static_element" for e in mesh.elements.values()):
            from ._ge_beam3_native_load_admission import assemble_nodal_forces

            return assemble_nodal_forces(
                self, mesh, dof_manager, guard=qualified_runtime_guard,
                activity=element_activity,
            )
        return self._get_load_vector_core(
            mesh,
            dof_manager,
            material_getter,
            displacements,
            element_activity,
            qualified_runtime_guard=qualified_runtime_guard,
        )

    def _get_load_vector_core(
        self,
        mesh: "FEMesh",
        dof_manager: "DOFManager",
        material_getter: Optional[Callable[[str], "Material"]] = None,
        displacements: Optional[np.ndarray] = None,
        element_activity: Optional[object] = None,
        *,
        qualified_runtime_guard: Callable[..., None],
    ) -> np.ndarray:
        """Assemble the global load vector once the element-family route is chosen."""
        total_dofs = dof_manager.total_dofs
        F = np.zeros(total_dofs)

        def activity_scale(element_id: int) -> float:
            if element_activity is None:
                return 1.0
            values = element_activity.load_scales([int(element_id)])
            qualified_runtime_guard(stage="load activity observation")
            return float(np.asarray(values, dtype=float).reshape(-1)[0])

        # Nodal loads.
        for node_id, load in self.nodal_loads.items():
            node = mesh.get_node(node_id)
            if node:
                for i, dof in enumerate(node.dofs):
                    if i < len(load):
                        F[dof] += load[i]

        # User-provided element load vectors.
        for element_id, load in self.element_loads.items():
            element = mesh.get_element(element_id)
            if element is None:
                continue
            dof_mapping = element.get_dof_mapping(mesh)
            load = np.asarray(load, dtype=float)
            scale = activity_scale(int(element_id))
            for i, dof in enumerate(dof_mapping):
                if i < len(load):
                    F[dof] += scale * load[i]

        # Consistent pressure loads for shell elements.
        for element_id, pressure in self.pressure_loads.items():
            element = mesh.get_element(element_id)
            if element is None or not hasattr(element, "node_ids"):
                continue
            coords = None
            if self.follower_pressure:
                self._reject_strict_flat_s3_v2_follower_pressure(element)
                coords = self._current_element_coordinates(element, mesh, displacements)
            f_elem = self._consistent_pressure_load(element, mesh, pressure, coords)
            scale = activity_scale(int(element_id))
            dof_mapping = element.get_dof_mapping(mesh)
            for i, dof in enumerate(dof_mapping):
                if i < len(f_elem):
                    F[dof] += scale * f_elem[i]

        # Gravity loads from element mass matrices.
        if self.gravity is not None:
            for element_id, element in mesh.elements.items():
                if not hasattr(element, "node_ids"):
                    continue
                if material_getter is None:
                    material = _GravityFallbackMaterial()
                else:
                    material = material_getter(element.material_name)
                    qualified_runtime_guard(
                        stage=(
                            "load material observation for element "
                            f"{element_id}"
                        )
                    )
                f_elem = self._consistent_gravity_load(element, mesh, material)
                qualified_runtime_guard(
                    stage=f"gravity-load element observation for element {element_id}"
                )
                scale = activity_scale(int(element_id))
                dof_mapping = element.get_dof_mapping(mesh)
                for i, dof in enumerate(dof_mapping):
                    if i < len(f_elem):
                        F[dof] += scale * f_elem[i]

        # Inertial load from added masses under the acceleration field: both
        # model-level point masses (which also enter the mass matrix) and any
        # load-case-only added masses.
        if self.gravity is not None:
            acceleration = np.asarray(self.gravity, dtype=float)
            from .concentrated_mass import entries
            for node, value in entries(mesh):
                F[np.asarray(node.dofs)] += value.matrix() @ np.r_[acceleration, np.zeros(3)]
            combined_masses: Dict[int, float] = {}
            for node_id, mass in getattr(mesh, "point_masses", {}).items():
                combined_masses[int(node_id)] = combined_masses.get(int(node_id), 0.0) + float(mass)
            for node_id, mass in self.added_node_masses.items():
                combined_masses[int(node_id)] = combined_masses.get(int(node_id), 0.0) + float(mass)
            for node_id, mass in combined_masses.items():
                node = mesh.get_node(int(node_id))
                if node is None or mass == 0.0:
                    continue
                for axis in range(3):
                    F[node.dofs[axis]] += float(mass) * acceleration[axis]

        return F

    def get_load_vector(
        self,
        mesh: "FEMesh",
        dof_manager: "DOFManager",
        material_getter: Optional[Callable[[str], "Material"]] = None,
        displacements: Optional[np.ndarray] = None,
        element_activity: Optional[object] = None,
    ) -> np.ndarray:
        """Assemble a direct load vector under one mesh-family authority lease."""

        return _run_with_qualified_load_runtime_lease(
            mesh,
            context="direct load-vector assembly",
            operation=lambda guard: self._get_load_vector_under_lease(
                mesh,
                dof_manager,
                material_getter,
                displacements,
                element_activity,
                qualified_runtime_guard=guard,
            ),
        )


def _callable_authority(member: Any, seen: tuple = ()) -> tuple:
    """Snapshot executable metadata, including descriptor accessors."""
    if type(member) is types.FunctionType:
        if any(member is prior for prior in seen):
            return (member,)
        closure = tuple(cell.cell_contents for cell in member.__closure__ or ())
        return (member, member.__code__, member.__defaults__,
                None if member.__kwdefaults__ is None else tuple(member.__kwdefaults__.items()),
                tuple((value, _callable_authority(value, seen + (member,)),
                       name == "accepted_generation" and type(value) is int
                       and member.__globals__.get("__name__") == "anysolver._qualified_authority_epoch"
                       and member.__qualname__ in (
                           "_AuthorityEpochManager.bind.<locals>.require",
                           "_AuthorityEpochManager.bind_context.<locals>.require",
                           "_AuthorityEpochManager.bind_argument.<locals>.require"))
                      for name, value in zip(member.__code__.co_freevars, closure)))
    if type(member) is property:
        return tuple(_callable_authority(f) for f in (member.fget, member.fset, member.fdel))
    if type(member) in (staticmethod, classmethod):
        return (_callable_authority(member.__func__),)
    return ()


def _callable_authority_matches(member: Any, authority: tuple) -> bool:
    if type(member) is types.FunctionType:
        if len(authority) == 1:
            return member is authority[0]
        if not authority or member is not authority[0] or member.__code__ is not authority[1] or member.__defaults__ is not authority[2]:
            return False
        closure = member.__closure__ or ()
        if len(closure) != len(authority[4]):
            return False
        try:
            if any((type(cell.cell_contents) is not int if generation_counter else
                    cell.cell_contents is not value or not _callable_authority_matches(value, captured))
                   for cell, (value, captured, generation_counter) in zip(closure, authority[4])):
                return False
        except ValueError:  # An emptied closure cell is no longer authoritative.
            return False
        expected = authority[3]
        current = member.__kwdefaults__
        return (current is None if expected is None else
                type(current) is dict and all(type(key) is str for key in current) and len(current) == len(expected) and
                all(current.get(key, _MISSING) is value for key, value in expected))
    if type(member) is property:
        return len(authority) == 3 and all(
            _callable_authority_matches(f, a)
            for f, a in zip((member.fget, member.fset, member.fdel), authority))
    if type(member) in (staticmethod, classmethod):
        return len(authority) == 1 and _callable_authority_matches(member.__func__, authority[0])
    return not authority


_MISSING = object()
_EXACT_LOAD_CASE_NAMESPACE = dict(type.__getattribute__(LoadCase, "__dict__"))
_EXACT_LOAD_CASE_GET_VECTOR_CORE = _EXACT_LOAD_CASE_NAMESPACE[
    "_get_load_vector_core"
]
_EXACT_FE_MESH_PROVIDERS = tuple(
    (name, type.__getattribute__(_FEMesh, "__dict__")[name])
    for name in ("get_node", "get_element")
)
_NATIVE_FIBRE_STATIC_MODULE = "anysolver._ge_beam3_native_fibre_static_element"
_EXACT_LOAD_CASE_FIELDS = frozenset(LoadCase.__dataclass_fields__)
_EXACT_LOAD_ROUTING_CLASSES = tuple(
    (owner, type.__getattribute__(owner, "__mro__"),
     dict(type.__getattribute__(owner, "__dict__")))
    for owner in (_FEModel, _FEMesh, _Node, _DOFManager, _QualifiedStateMapping, ABCMeta)
)

_EXACT_LOAD_CALLABLE_METADATA = {
    owner: {name: _callable_authority(member) for name, member in namespace.items()}
    for owner, _mro, namespace in _EXACT_LOAD_ROUTING_CLASSES
}
_EXACT_LOAD_CASE_CALLABLE_METADATA = {
    name: _callable_authority(member) for name, member in _EXACT_LOAD_CASE_NAMESPACE.items()
}
_EXACT_MASS_CLASSES = tuple(
    (owner, type.__getattribute__(owner, "__mro__"),
     dict(type.__getattribute__(owner, "__dict__")))
    for owner in (_BeamElement, _Element, _ShellElement, _IsotropicMaterial, _AuthorityEpochManager,
                  _CoupledBeamShellElement,
                  _QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement)
)
_EXACT_MASS_METADATA = {
    owner: {name: _callable_authority(member) for name, member in namespace.items()}
    for owner, _mro, namespace in _EXACT_MASS_CLASSES
}
_WARNING_FUNCTIONS = tuple(
    (name, getattr(warnings, name), _callable_authority(getattr(warnings, name)))
    for name in ("showwarning", "formatwarning", "_showwarnmsg", "_showwarnmsg_impl",
                 "_formatwarnmsg", "_formatwarnmsg_impl")
)


_WARNING_MESSAGE_CLASS = warnings.WarningMessage
_WARNING_MESSAGE_NAMESPACE = dict(vars(_WARNING_MESSAGE_CLASS))
_NUMPY_GETERR = _NUMPY_ERROR_MODULE.geterr
_NUMPY_GETERR_AUTHORITY = _callable_authority(_NUMPY_GETERR)
if sys.version_info >= (3, 14):
    import _py_warnings as _WARNING_IMPLEMENTATION
else:
    _WARNING_IMPLEMENTATION = warnings


class _InstanceDictLayoutProbe:
    pass


def _shared_instance_dict_descriptor() -> Any:
    """CPython 3.15's one ``__dict__`` getset descriptor shared by all classes.

    Up to 3.14 each class with an instance dict gets its own descriptor, owned
    (``__objclass__``) by that class. From 3.15 every class stores the same
    descriptor, owned by ``object``. Captured once at import from a fresh class;
    None where the interpreter does not use the shared layout.
    """
    if sys.version_info < (3, 15):
        return None
    descriptor = type.__getattribute__(_InstanceDictLayoutProbe, "__dict__").get("__dict__")
    if (type(descriptor) is not types.GetSetDescriptorType
            or descriptor.__objclass__ is not object
            or descriptor.__name__ != "__dict__"):
        return None
    return descriptor


_SHARED_INSTANCE_DICT = _shared_instance_dict_descriptor()


def _builtin_instance_dict_descriptor(descriptor: Any, owner: type) -> bool:
    """True for the interpreter's own ``__dict__`` descriptor of ``owner``.

    That is the per-class getset descriptor owned by ``owner`` (3.14 and
    earlier), or exactly the shared descriptor captured at import (3.15).
    """
    return (type(descriptor) is types.GetSetDescriptorType
            and descriptor.__name__ == "__dict__"
            and (descriptor.__objclass__ is owner
                 or (_SHARED_INSTANCE_DICT is not None and descriptor is _SHARED_INSTANCE_DICT)))


def _canonical_python_function(function: Any, module: Any, name: str) -> bool:
    # Python 3.14's warnings facade re-exports the stdlib implementation.
    if module is warnings:
        module = _WARNING_IMPLEMENTATION
    return (type(function) is types.FunctionType and function.__globals__ is vars(module)
            and (function.__code__.co_filename == module.__file__ or
                 (module.__spec__.origin == "frozen" and
                  function.__code__.co_filename == "<frozen " + module.__name__ + ">"))
            and function.__code__.co_name == name)


_WARNING_CANONICAL_ORIGIN = {
    name: _canonical_python_function(function, warnings, name)
    for name, function, _authority in _WARNING_FUNCTIONS
}
if _WARNING_IMPLEMENTATION is not warnings:
    _WARNING_GET_CONTEXT = _WARNING_IMPLEMENTATION._get_context
    _WARNING_GET_CONTEXT_AUTHORITY = _callable_authority(_WARNING_GET_CONTEXT)
    _WARNING_GLOBAL_CONTEXT = _WARNING_IMPLEMENTATION._global_context
    _WARNING_GLOBAL_CLASS = type(_WARNING_GLOBAL_CONTEXT)
    _WARNING_CONTEXT_CLASSES = tuple(
        (owner, type.__getattribute__(owner, "__mro__"), dict(vars(owner)),
         {name: _callable_authority(member) for name, member in vars(owner).items()})
        for owner in _WARNING_GLOBAL_CLASS.__mro__)
_WARNING_MESSAGE_METADATA = {name: _callable_authority(member)
                             for name, member in _WARNING_MESSAGE_NAMESPACE.items()}
_MASS_HELPERS = tuple((name, getattr(_element_module, name),
                       _callable_authority(getattr(_element_module, name)))
                      for name in ("_beam_rotation_matrix", "_cross3"))
_CODECS_LOOKUP_ERROR = codecs.lookup_error
_CHARMAP_BUILD = codecs.charmap_build
_WARNING_CHARMAP_TYPE = (
    type(_CHARMAP_BUILD("".join(chr(value) for value in range(256))))
    if type(_CHARMAP_BUILD) is types.BuiltinFunctionType and _CHARMAP_BUILD.__name__ == "charmap_build"
    else None
)
_BUILTIN_WARNING_ERROR_HANDLERS = {
    "strict": codecs.strict_errors,
    "backslashreplace": codecs.backslashreplace_errors,
}
_LINECACHE_FUNCTIONS = tuple((name, getattr(linecache, name),
                            _callable_authority(getattr(linecache, name)))
                           for name in ("getline", "getlines", "updatecache"))


_WARNING_CODECS = {
    "utf8": (_UTF8_CODEC, "utf_8_encode"),
    "ascii": (_ASCII_CODEC, "ascii_encode"),
    "latin1": (_LATIN1_CODEC, "latin_1_encode"),
    "iso88591": (_LATIN1_CODEC, "latin_1_encode"),
    "cp1252": (_CP1252_CODEC, "charmap_encode"),
}
_WARNING_ENCODER_AUTHORITY = {
    module: (
        module.IncrementalEncoder,
        tuple((owner, type.__getattribute__(owner, "__mro__"),
               dict(type.__getattribute__(owner, "__dict__")),
               {name: _callable_authority(member)
                for name, member in type.__getattribute__(owner, "__dict__").items()})
              for owner in (module.IncrementalEncoder, codecs.IncrementalEncoder)),
        getattr(codecs, helper),
        getattr(module, "encoding_table", None),
    )
    for module, helper in _WARNING_CODECS.values()
}


def _referent_provider_is_builtin() -> bool:
    return (type(_GET_REFERENTS) is types.BuiltinFunctionType
            and _GET_REFERENTS.__module__ == "gc" and _GET_REFERENTS.__name__ == "get_referents")


def _builtin_warning_encoder(stream: Any) -> bool:
    if not _referent_provider_is_builtin():
        return False
    encoding = stream.encoding.lower().replace("-", "")
    if encoding not in _WARNING_CODECS:
        return False
    module, helper = _WARNING_CODECS[encoding]
    encoder_class, classes, native, encoding_table = _WARNING_ENCODER_AUTHORITY[module]
    base = classes[1][0]
    bases = type.__getattribute__(encoder_class, "__bases__")
    if (module.IncrementalEncoder is not encoder_class
            or codecs.IncrementalEncoder is not base or len(bases) != 1 or bases[0] is not base
            or not _canonical_python_function(vars(base).get("__init__"), codecs, "__init__")
            or not _canonical_python_function(vars(encoder_class).get("encode"), module, "encode")
            or type(native) is not types.BuiltinFunctionType
            or native.__name__ != helper
            or getattr(codecs, helper) is not native
            or getattr(module, "encoding_table", None) is not encoding_table
            or (module is _CP1252_CODEC and type(encoding_table) is not _WARNING_CHARMAP_TYPE)):
        return False
    for owner, mro, namespace, metadata in classes:
        current = type.__getattribute__(owner, "__dict__")
        if (type.__getattribute__(owner, "__mro__") != mro or len(current) != len(namespace)
                or any(current.get(name, _MISSING) is not value or
                       not _callable_authority_matches(value, metadata[name])
                       for name, value in namespace.items())):
            return False
    # Inspect the actual encoder retained by TextIOWrapper; looking up a codec
    # by name could run a registered foreign search function and prove a
    # different encoder from the one this stream will actually call.
    encoders = [value for value in _GET_REFERENTS(stream) if type(value) is encoder_class]
    if len(encoders) != 1:
        return False
    state = object.__getattribute__(encoders[0], "__dict__")
    return (type(state) is dict and all(type(key) is str for key in state)
            and set(state).issubset({"errors", "buffer"})
            and type(state.get("errors")) is str and state["errors"] == stream.errors
            and ("buffer" not in state or type(state["buffer"]) is str))


def _builtin_warning_stream(stream: Any) -> bool:
    if type(stream) is not io.TextIOWrapper or any(name in vars(stream) for name in ("write", "flush")):
        return False
    if not _builtin_warning_encoder(stream) or stream.errors not in _BUILTIN_WARNING_ERROR_HANDLERS:
        return False
    handler = _BUILTIN_WARNING_ERROR_HANDLERS[stream.errors]
    if (type(_CODECS_LOOKUP_ERROR) is not types.BuiltinFunctionType
            or _CODECS_LOOKUP_ERROR.__module__ != "_codecs"
            or _CODECS_LOOKUP_ERROR.__name__ != "lookup_error"
            or type(handler) is not types.BuiltinFunctionType
            or handler.__name__ != stream.errors + "_errors"
            or _CODECS_LOOKUP_ERROR(stream.errors) is not handler):
        return False
    buffer = stream.buffer
    if type(buffer) is io.FileIO:
        return not any(name in vars(buffer) for name in ("write", "flush"))
    if type(buffer) not in (io.BufferedWriter, io.BufferedRandom):
        return False
    raw = buffer.raw
    return (not any(name in vars(buffer) for name in ("write", "flush"))
            and type(raw) is io.FileIO
            and not any(name in vars(raw) for name in ("write", "flush")))


def _numpy_error_routes_are_builtin() -> bool:
    """Do not enter an owned span that can dispatch to foreign error handlers.

    No error policy is changed. Warn/print remain eligible only with the
    standard warning machinery, plain filters, and builtin output streams.
    Capture/custom handlers fall back to the public guarded route.
    """
    if _WARNING_IMPLEMENTATION is not warnings:
        implementation = vars(_WARNING_IMPLEMENTATION)
        # Context-aware warning routing is deliberately left on the full path.
        # Authenticate the ordinary context without evaluating a foreign getter.
        use_context = implementation.get("_use_context")
        if (implementation.get("_wm") is not warnings
                or not (use_context is False or (type(use_context) is int and use_context == 0))
                or vars(warnings).get("_get_context") is not _WARNING_GET_CONTEXT
                or not _canonical_python_function(_WARNING_GET_CONTEXT, warnings, "_get_context")
                or not _callable_authority_matches(_WARNING_GET_CONTEXT, _WARNING_GET_CONTEXT_AUTHORITY)
                or implementation.get("_global_context") is not _WARNING_GLOBAL_CONTEXT
                or type(_WARNING_GLOBAL_CONTEXT) is not _WARNING_GLOBAL_CLASS
                or type.__getattribute__(_WARNING_GLOBAL_CLASS, "__getattribute__") is not object.__getattribute__):
            return False
        for owner, mro, namespace, metadata in _WARNING_CONTEXT_CLASSES:
            current = vars(owner)
            dictionary = current.get("__dict__", _MISSING)
            if (type(owner) is not type or "log" in current
                    or (dictionary is not _MISSING
                        and not _builtin_instance_dict_descriptor(dictionary, owner))
                    or type.__getattribute__(owner, "__mro__") != mro
                    or len(current) != len(namespace)
                    or any(current.get(name, _MISSING) is not member or
                           not _callable_authority_matches(member, metadata[name])
                           for name, member in namespace.items())):
                return False
        context = object.__getattribute__(_WARNING_GLOBAL_CONTEXT, "__dict__")
        if (type(context) is not dict or len(context) != 1
                or not all(type(key) is str for key in context)
                or context.get("log", _MISSING) is not None):
            return False
    if (np.geterr is not _NUMPY_GETERR
            or type(_NUMPY_GETERR) is not types.FunctionType
            or _NUMPY_GETERR.__globals__ is not vars(_NUMPY_ERROR_MODULE)
            or _NUMPY_GETERR.__code__.co_filename != _NUMPY_ERROR_MODULE.__file__
            or _NUMPY_GETERR.__code__.co_name != "geterr"
            or not _callable_authority_matches(_NUMPY_GETERR, _NUMPY_GETERR_AUTHORITY)):
        return False
    modes = tuple(_NUMPY_GETERR().values())
    if any(mode in ("call", "log") for mode in modes):
        return False
    if "print" in modes and not _builtin_warning_stream(sys.stderr):
        return False
    if "warn" not in modes:
        return True
    # Pytest and catch_warnings(record=True) use an exact builtin list.append.
    # This captures a builtin WarningMessage without formatting/output hooks.
    if warnings.showwarning is not warnings._showwarning_orig:
        return False
    implementation = warnings._showwarnmsg_impl
    captured = (type(implementation) is types.BuiltinMethodType
                and type(implementation.__self__) is list
                and implementation.__name__ == "append")
    for name, function, authority in _WARNING_FUNCTIONS:
        if captured and name in ("_showwarnmsg_impl", "_formatwarnmsg", "_formatwarnmsg_impl", "formatwarning"):
            continue
        if not _WARNING_CANONICAL_ORIGIN[name] or getattr(warnings, name) is not function or not _callable_authority_matches(function, authority):
            return False
    if warnings.WarningMessage is not _WARNING_MESSAGE_CLASS or type(warnings.WarningMessage) is not type:
        return False
    if not _canonical_python_function(_WARNING_MESSAGE_NAMESPACE.get("__init__"), warnings, "__init__"):
        return False
    if not _canonical_python_function(_WARNING_MESSAGE_NAMESPACE.get("__str__"), warnings, "__str__"):
        return False
    message_namespace = vars(_WARNING_MESSAGE_CLASS)
    if len(message_namespace) != len(_WARNING_MESSAGE_NAMESPACE) or any(
        message_namespace.get(name, _MISSING) is not member or
        not _callable_authority_matches(member, _WARNING_MESSAGE_METADATA[name])
        for name, member in _WARNING_MESSAGE_NAMESPACE.items()
    ):
        return False
    if not captured:
        if warnings.formatwarning is not warnings._formatwarning_orig or not _builtin_warning_stream(sys.stderr):
            return False
        if any(not _canonical_python_function(function, linecache, name)
               or getattr(linecache, name) is not function
               or not _callable_authority_matches(function, authority)
               for name, function, authority in _LINECACHE_FUNCTIONS):
            return False
    if type(warnings.filters) is not list:
        return False
    for entry in warnings.filters:
        if type(entry) is not tuple or len(entry) != 5:
            return False
        action, message, category, module, lineno = entry
        if (type(action) is not str or type(lineno) is not int
                or (message is not None and type(message) not in (str, re.Pattern))
                or (module is not None and type(module) not in (str, re.Pattern))
                or type(category) is not type):
            return False
    return True


def _plain_mass_input(value: Any, depth: int = 0) -> bool:
    """Fail closed on foreign values, including nested section/cache data."""
    if depth > 12:
        return False
    if value is None or type(value) in (bool, int, float, str, bytes, np.float64, np.float32, np.int64, np.int32):
        return True
    if _plain_numeric_array(value):
        return True
    if type(value) in (list, tuple):
        return all(_plain_mass_input(item, depth + 1) for item in value)
    if type(value) is types.MappingProxyType:
        if not _referent_provider_is_builtin():
            return False
        # A proxy can wrap a foreign Mapping; never call its mapping methods.
        referents = _GET_REFERENTS(value)
        if len(referents) != 1 or type(referents[0]) is not dict:
            return False
        value = referents[0]
    if type(value) is dict:
        return all(type(key) in (str, int) and _plain_mass_input(item, depth + 1)
                   for key, item in dict.items(value))
    return False


def _capture_mass_function_dependencies() -> tuple:
    pending = [owner.__dict__[name] for owner, name in (
        (_BeamElement, "compute_mass_matrix"), (_BeamElement, "_beam_frame_and_transform"),
        (_QualifiedE4PLShellElement, "compute_mass_matrix"),
        (_QualifiedE4PLS3ShellElement, "compute_mass_matrix"),
        (_QualifiedE4PLS3ShellElement, "_compute_mass_components"),
        (_QualifiedE4PLS3ShellElement, "_compute_stiffness_components"),
        (_CoupledBeamShellElement, "compute_mass_matrix"))]
    seen = set()
    dependencies = []
    methods = {}
    for owner in (_BeamElement, _Element, _ShellElement, _IsotropicMaterial,
                  _CoupledBeamShellElement,
                  _QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement,
                  _AuthorityEpochManager, _Node, _FEMesh, _DOFManager):
        for name, member in type.__getattribute__(owner, "__dict__").items():
            functions = ((member.fget, member.fset, member.fdel) if type(member) is property else
                         (member.__func__,) if type(member) in (classmethod, staticmethod) else (member,))
            methods.setdefault(name, []).extend(function for function in functions
                                                if type(function) is types.FunctionType)
    while pending:
        function = pending.pop()
        if type(function) is not types.FunctionType or id(function) in seen:
            continue
        seen.add(id(function))
        for cell in function.__closure__ or ():
            value = cell.cell_contents
            if type(value) is types.FunctionType:
                pending.append(value)
        codes = [function.__code__]
        names = set()
        while codes:
            code = codes.pop()
            names.update(code.co_names)
            codes.extend(value for value in code.co_consts if type(value) is types.CodeType)
        namespace = function.__globals__
        for name in names:
            pending.extend(methods.get(name, ()))
            value = namespace.get(name)
            if (type(value) is types.FunctionType and
                    value.__globals__.get("__name__", "").startswith(("anysolver.", "anymaterial."))):
                dependencies.append((namespace, name, value, _callable_authority(value)))
                pending.append(value)
    return tuple(dependencies)


_EXACT_MASS_FUNCTION_DEPENDENCIES = _capture_mass_function_dependencies()


def _mass_classes_are_untouched() -> bool:
    if any(namespace.get(name, _MISSING) is not function or
           not _callable_authority_matches(function, authority)
           for namespace, name, function, authority in _EXACT_MASS_FUNCTION_DEPENDENCIES):
        return False
    if any(getattr(_element_module, name) is not function or
           not _callable_authority_matches(function, authority)
           for name, function, authority in _MASS_HELPERS):
        return False
    for owner, mro, expected in _EXACT_MASS_CLASSES:
        if type.__getattribute__(owner, "__mro__") != mro:
            return False
        current = type.__getattribute__(owner, "__dict__")
        extra = 1 if "__slotnames__" in current and "__slotnames__" not in expected else 0
        if len(current) - extra != len(expected):
            return False
        if any(current.get(name, _MISSING) is not member or
               not _callable_authority_matches(member, _EXACT_MASS_METADATA[owner][name])
               for name, member in expected.items()):
            return False
    return True


def _plain_coupled_element_state(data: dict) -> bool:
    """Exact coupling state: plain ids/stiffness and a numeric eccentricity.

    The acceleration span never evaluates this state (a coupling element's
    mass matrix is exactly zero), but admitting only constructed plain
    state keeps a hand-crafted instance from smuggling an executable value
    past the exact-type and class-authority checks.
    """
    return (
        type(data.get("beam_node_id")) is int
        and type(data.get("shell_node_id")) is int
        and type(data.get("coupling_stiffness")) in (float, int)
        and (data.get("eccentricity") is None
             or _plain_numeric_array(data.get("eccentricity")))
    )


def _builtin_acceleration_inputs(state: dict, mesh_state: dict, material_getter: Any,
                                 elements: Any, loaded_nodes: set) -> bool:
    if mesh_state.get("concentrated_masses"):
        return False  # preserve the public validated mass/load observation route
    acceleration = state.get("gravity")
    if not _plain_numeric_array(acceleration) or not _mass_classes_are_untouched():
        return False
    # Only the ordinary FEModel provider and exact canonical isotropic material
    # are admitted. Generalized/laminate sections retain their public route.
    if (type(material_getter) is not types.MethodType
            or material_getter.__func__ is not _FEModel.get_material
            or not _load_model_access_is_builtin(material_getter.__self__)
            or object.__getattribute__(object.__getattribute__(material_getter.__self__, "__dict__")["mesh"], "__dict__") is not mesh_state):
        return False
    model_state = object.__getattribute__(material_getter.__self__, "__dict__")
    if type(model_state.get("materials")) is not dict or type(model_state.get("current_material")) not in (str, type(None)):
        return False
    materials = model_state["materials"]
    if not all(type(key) is str for key in materials) or "default" not in materials:
        return False
    for material in materials.values():
        if type(material) is not _IsotropicMaterial or not _plain_mass_input(object.__getattribute__(material, "__dict__")):
            return False
    for mapping in (mesh_state.get("point_masses", {}), state.get("added_node_masses")):
        if type(mapping) is not dict or not all(type(key) is int and type(mass) in (float, int) for key, mass in mapping.items()):
            return False
        loaded_nodes.update(mapping)
    # Epoch tokens/component guards are checked by the full lease before this
    # span; they are package-owned authority objects, never numerical inputs.
    authority_fields = {"_qualified_direct_state_tokens", "_qualified_direct_state_token",
                        "_qualified_component_guard"}
    mass_class_names = {
        kind: {name for owner in type.__getattribute__(kind, "__mro__")
               for name in type.__getattribute__(owner, "__dict__")}
        for kind in (_BeamElement, _CoupledBeamShellElement,
                     _QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement)
    }
    for element in dict.values(elements):
        if type(element) not in (_BeamElement, _CoupledBeamShellElement,
                                 _QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement):
            return False
        data = object.__getattribute__(element, "__dict__")
        if type(data) is not dict or not all(type(key) is str for key in data):
            return False
        if any(key in mass_class_names[type(element)] for key in data):
            return False
        if data.get("shell_section") is not None or data.get("generalized_section") is not None:
            return False
        if not all(key in authority_fields or _plain_mass_input(value) for key, value in data.items()):
            return False
        if type(element) is _CoupledBeamShellElement and not _plain_coupled_element_state(data):
            return False
        ids = data.get("node_ids")
        if type(ids) not in (list, tuple) or not all(type(node_id) is int for node_id in ids) or type(data.get("material_name")) is not str:
            return False
        loaded_nodes.update(ids)
    return True


def _plain_numeric_array(value: Any) -> bool:
    """Exact arrays with builtin numeric scalars cannot invoke object hooks."""
    return (type(value) is np.ndarray and value.dtype.kind in "biuf"
            and (value.dtype.kind != "f" or value.dtype.itemsize <= 8))


def _load_routing_classes_are_untouched() -> bool:
    for owner, mro, expected in _EXACT_LOAD_ROUTING_CLASSES:
        if type(owner) is not type or type.__getattribute__(owner, "__mro__") != mro:
            return False
        current = type.__getattribute__(owner, "__dict__")
        extra = 1 if "__slotnames__" in current and "__slotnames__" not in expected else 0
        if len(current) - extra != len(expected) or any(
            current.get(name, _MISSING) is not member or
            not _callable_authority_matches(member, _EXACT_LOAD_CALLABLE_METADATA[owner][name])
            for name, member in expected.items()
        ):
            return False
    return True


def _load_model_access_is_builtin(model: Any) -> bool:
    """Prove the ordinary model/mesh/DOF reads surrounding the kernel are owned."""
    if type(model) is not _FEModel or not _load_routing_classes_are_untouched():
        return False
    state = object.__getattribute__(model, "__dict__")
    if type(state) is not dict or not all(type(key) is str for key in state):
        return False
    mesh = state.get("mesh")
    if "get_material" in state or type(mesh) is not _FEMesh:
        return False
    mesh_state = object.__getattribute__(mesh, "__dict__")
    if type(mesh_state) is not dict or not all(type(key) is str for key in mesh_state):
        return False
    manager = mesh_state.get("dof_manager")
    if (
        type(manager) is not _DOFManager
        or mesh_state.get("element_activity") is not None
        or "num_nodes" in mesh_state
        or not _mapping_reads_are_builtin(mesh_state.get("nodes"))
    ):
        return False
    manager_state = object.__getattribute__(manager, "__dict__")
    return (
        type(manager_state) is dict
        and all(type(key) is str for key in manager_state)
        and type(manager_state.get("_total_dofs")) is int
    )


def _exact_load_case_instance_namespace(load_case: Any) -> Optional[dict]:
    """Return the plain instance namespace of an untouched ``LoadCase``.

    ``None`` means attribute observation could run foreign code: the object is
    not exactly a ``LoadCase``, its class namespace differs from the one this
    module defined (a replaced method, a data descriptor shadowing a field, a
    ``__getattr__``), or its instance state is not the plain dataclass state.
    Nothing here executes user code.
    """

    if type(load_case) is not LoadCase or type(LoadCase) is not type:
        return None
    mro = type.__getattribute__(LoadCase, "__mro__")
    if len(mro) != 2 or mro[1] is not object:
        return None
    namespace = type.__getattribute__(LoadCase, "__dict__")
    exact = _EXACT_LOAD_CASE_NAMESPACE
    # ``copy``/``pickle`` cache ``__slotnames__`` on the class; it is data only
    # and can never shadow a field or hook attribute access.
    extra = 1 if "__slotnames__" in namespace and "__slotnames__" not in exact else 0
    if len(namespace) - extra != len(exact):
        return None
    for key, expected in exact.items():
        if (namespace.get(key, _MISSING) is not expected or
                not _callable_authority_matches(expected, _EXACT_LOAD_CASE_CALLABLE_METADATA[key])):
            return None
    state = object.__getattribute__(load_case, "__dict__")
    if (
        type(state) is not dict
        or not all(type(key) is str for key in state)
        or not set(state).issubset(_EXACT_LOAD_CASE_FIELDS)
        or type(state.get("name")) is not str
        or type(state.get("follower_pressure")) is not bool
        or type(state.get("nodal_loads")) is not dict
        or type(state.get("element_loads")) is not dict
        or type(state.get("pressure_loads")) is not dict
    ):
        return None
    return state


def _mapping_reads_are_builtin(mapping: Any) -> bool:
    """True for a plain dict or the package's state mapping with ``dict.get``."""

    kind = type(mapping)
    if kind is dict:
        return True
    if kind is not _QualifiedStateMapping:
        return False
    if (
        type.__getattribute__(_QualifiedStateMapping, "__dict__").get("get")
        is not None
        or type.__getattribute__(_QualifiedStateMapping, "__mro__")[1] is not dict
    ):
        return False
    state = object.__getattribute__(mapping, "__dict__")
    return type(state) is dict and all(type(key) is str for key in state) and "get" not in state


def _run_foreign_code_free_load_vector(
    load_case: Any,
    mesh: Any,
    dof_manager: Any,
    material_getter: Optional[Callable[[str], "Material"]],
    displacements: Optional[np.ndarray],
    element_activity: Optional[object],
    *,
    stage_guard: Callable[..., None],
) -> Optional[np.ndarray]:
    """Assemble a load vector whose whole evaluation runs only owned code.

    Returns ``None`` when that cannot be established, and the caller then uses
    the ordinary public path.  The evaluation is foreign-code-free when the
    load case is an untouched ``LoadCase`` with only nodal, element and
    pressure/acceleration loads (no activity provider), every loaded element is
    an exact qualified Q4/S3 element (their class authority is epoch-tracked by
    the lease), every element contributing acceleration load is an exact
    package beam or beam-shell coupling element with untouched class
    authority and plain constructed state, the mesh is an exact ``FEMesh``
    with its providers unchanged,
    every value read is a plain ``float``/``int``/``tuple``/``ndarray`` and no
    mesh element belongs to the native fibre family that takes a different
    route. Cheap builtin family/key scans and loaded-entry checks avoid repeated
    model-wide lifecycle/lease validation; a span it admits
    can then be bracketed by the lease's constant-time trusted guard because
    no code that could change model state runs inside it.
    """

    if element_activity is not None or not _numpy_error_routes_are_builtin():
        return None
    state = _exact_load_case_instance_namespace(load_case)
    if state is None:
        return None
    if displacements is not None and not _plain_numeric_array(displacements):
        return None
    if type(mesh) is not _FEMesh or not _load_routing_classes_are_untouched():
        return None
    mesh_class = type.__getattribute__(_FEMesh, "__dict__")
    for name, expected in _EXACT_FE_MESH_PROVIDERS:
        if mesh_class.get(name, _MISSING) is not expected:
            return None
    mesh_state = object.__getattribute__(mesh, "__dict__")
    if type(mesh_state) is not dict or not all(type(key) is str for key in mesh_state):
        return None
    elements = mesh_state.get("elements")
    nodes = mesh_state.get("nodes")
    if (
        not _mapping_reads_are_builtin(elements)
        or not _mapping_reads_are_builtin(nodes)
        or "get_node" in mesh_state
        or "get_element" in mesh_state
        or type(dof_manager) is not _DOFManager
    ):
        return None
    if not all(type(key) is int for mapping in (nodes, elements) for key in dict.keys(mapping)):
        return None
    loaded_nodes = set()
    if state.get("gravity") is not None and not _builtin_acceleration_inputs(
        state, mesh_state, material_getter, elements, loaded_nodes
    ):
        return None
    for element in dict.values(elements):
        if type(element) in (_QualifiedE4PLShellElement, _QualifiedE4PLS3ShellElement):
            continue  # Owned epoch-tracked metaclasses; never the fibre family.
        if type(type(element)) not in (type, ABCMeta):
            return None
        module = type.__getattribute__(type(element), "__dict__").get("__module__")
        if (
            type(module) is not str or module == _NATIVE_FIBRE_STATIC_MODULE
        ):
            return None
    for load in state["nodal_loads"].values():
        if not _plain_numeric_array(load):
            return None
    for load in state["element_loads"].values():
        if not _plain_numeric_array(load):
            return None
    for source in (state["element_loads"], state["pressure_loads"]):
        for element_id in source:
            if type(element_id) is not int:
                return None
            element = dict.get(elements, element_id)
            if element is not None and type(element) not in (
                _QualifiedE4PLShellElement,
                _QualifiedE4PLS3ShellElement,
            ):
                return None
            if element is not None:
                element_state = object.__getattribute__(element, "__dict__")
                node_ids = dict.get(element_state, "node_ids")
                if type(node_ids) not in (list, tuple) or not all(
                    type(node_id) is int for node_id in node_ids
                ):
                    return None
                loaded_nodes.update(node_ids)
    for pressure in state["pressure_loads"].values():
        if type(pressure) in (float, int):
            continue
        if type(pressure) is not tuple or not all(
            type(value) is float for value in pressure
        ):
            return None
    for node_id in state["nodal_loads"]:
        if type(node_id) is not int:
            return None
        loaded_nodes.add(node_id)
    for node_id in loaded_nodes:
        node = dict.get(nodes, node_id)
        if node is None:
            continue
        if type(node) is not _Node:
            return None
        node_state = object.__getattribute__(node, "__dict__")
        if type(node_state) is not dict or not all(type(key) is str for key in node_state):
            return None
        dofs = node_state.get("dofs")
        if (
            "coords" in node_state
            or type(dofs) is not list
            or not all(type(dof) is int for dof in dofs)
            or not all(type(node_state.get(axis)) in (float, int) for axis in ("x", "y", "z"))
        ):
            return None
    stage_guard(stage="load-vector preflight")
    return _EXACT_LOAD_CASE_GET_VECTOR_CORE(
        load_case,
        mesh,
        dof_manager,
        material_getter,
        displacements,
        element_activity,
        qualified_runtime_guard=stage_guard,
    )


@dataclass
class InPlaneLoad:
    """
    In-plane load for stiffened panels.

    Represents axial, transverse and shear stresses applied to the panel.
    """

    axial_stress: float = 0.0
    transverse_stress: float = 0.0
    shear_stress: float = 0.0
    pressure: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "axial_stress": self.axial_stress,
            "transverse_stress": self.transverse_stress,
            "shear_stress": self.shear_stress,
            "pressure": self.pressure,
        }


class LoadCombination:
    """Linear combination of load cases."""

    def __init__(self, name: str, factors: Dict[str, float]):
        self.name = name
        self.factors = factors

    def get_combined_load_vector(
        self,
        load_cases: List[LoadCase],
        mesh: "FEMesh",
        dof_manager: "DOFManager",
        material_getter: Optional[Callable[[str], "Material"]] = None,
        element_activity: Optional[object] = None,
    ) -> np.ndarray:
        """Assemble the factored load vector.

        Pass ``model.get_material`` when combinations include gravity so each
        element uses its assigned density.
        """
        F_total = np.zeros(dof_manager.total_dofs)
        for load_case in load_cases:
            if load_case.name in self.factors:
                factor = self.factors[load_case.name]
                F_total += factor * load_case.get_load_vector(
                    mesh,
                    dof_manager,
                    material_getter,
                    element_activity=element_activity,
                )
        return F_total


# Common boundary condition factory functions

def create_fixed_support(name: str, node_ids: List[int]) -> FixedSupport:
    """Create a fixed support boundary condition."""
    return FixedSupport(name, node_ids)


def create_pinned_support(name: str, node_ids: List[int]) -> PinnedSupport:
    """Create a pinned support boundary condition."""
    return PinnedSupport(name, node_ids)


def create_roller_support(
    name: str,
    node_ids: List[int],
    constrained_directions: Optional[List[str]] = None,
) -> RollerSupport:
    """Create a roller support boundary condition."""
    return RollerSupport(name, node_ids, constrained_directions)


def create_symmetry_bc(name: str, node_ids: List[int], symmetry_plane: str = "xy") -> SymmetryBC:
    """Create a symmetry boundary condition."""
    return SymmetryBC(name, node_ids, symmetry_plane)
