"""
Core Finite Element Classes

This module contains the fundamental classes for FE analysis:
- DOFManager: Manages degrees of freedom and numbering
- FEMesh: Stores nodes, elements, and connectivity
- FEModel: Complete FE model with materials, loads, and results
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np

from anymaterial import IsotropicMaterial as Material

from .materials import (
    Hill48Yield,
    OrthotropicMaterial,
    StructuralMaterial,
    validate_material,
)

if TYPE_CHECKING:
    from .elements import Element
    from .boundary import BoundaryCondition, LoadCase
    from .constraint_audit import ConstraintEquation


_ELEMENT_LOCAL_CACHE_NAMES = (
    "_stiffness_matrix",
    "_mass_matrix",
    "_internal_forces",
    "_nl_cache",
    "_nl_cache_key",
    "_qualified_component_guard",
    "_hourglass_stiffness_matrix",
    "_qualified_components",
    "_qualified_cache_key",
)


def _clear_element_local_caches(element: "Element") -> None:
    """Clear mesh/material-dependent state on one element in constant time."""

    for name in _ELEMENT_LOCAL_CACHE_NAMES:
        if hasattr(element, name):
            # Derived caches are solver-owned.  Qualified element classes
            # reject ordinary post-construction writes to these names so an
            # external caller cannot inject stale mechanics; lifecycle
            # invalidation therefore uses the explicit internal boundary.
            object.__setattr__(element, name, None)


def _is_canonical_qualified_vector(value: Any) -> bool:
    """Return whether ``value`` already has the qualified frozen-vector form.

    Qualified vector inputs are stored as contiguous, aligned, native-
    ``float64`` 3-vectors without dtype metadata, backed by immutable
    bytes.  Recreating that same view is value-preserving, but it changes
    its object identity.  Recovery plan preparation reuses warm assembly
    plans that bind the identity of their immutable inputs, so needless
    recreation during that preparation discards reusable work.  Recognise
    the canonical representation before allocating a replacement; a
    readonly flag alone does not prove immutable backing, and metadata-
    bearing or unaligned views keep the original normalization outcome.
    """

    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.dtype.metadata is not None
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not value.flags.aligned
    ):
        return False
    current: Any = value
    seen: set[int] = set()
    while type(current) is np.ndarray:
        if current.flags.writeable or id(current) in seen:
            return False
        seen.add(id(current))
        current = current.base
        if current is None:
            return False
    return bool(
        type(current) is bytes
        or (
            type(current) is memoryview
            and current.readonly
            and type(current.obj) is bytes
        )
    )


def _freeze_qualified_element_vector_inputs(element: "Element") -> None:
    """Restore immutable vector inputs for qualified cache-aware elements.

    Recovery plan preparation and the mesh deepcopy backstop call this to
    restore the frozen input form.  The operation is idempotent for an
    already canonical immutable input, so it retains the object identity
    bound by existing warm assembly plans during that preparation.  Public
    mapping reinsertion advances the mutation epoch and deepcopy drops
    derived caches; both deliberately invalidate warm plans, and this
    helper does not preserve them across those operations.
    """

    if not hasattr(element, "_qualified_plan_state_revision"):
        return
    for name in ("material_direction", "reference_normal"):
        value = getattr(element, name, None)
        if value is None:
            continue
        if _is_canonical_qualified_vector(value):
            continue
        made = np.ascontiguousarray(np.asarray(value, dtype=float))
        frozen = np.frombuffer(made.tobytes(order="C"), dtype=float).reshape(
            made.shape
        )
        object.__setattr__(element, name, frozen)


class _QualifiedMutationEpoch(list[int]):
    """One-cell monotonic epoch with the legacy list read interface.

    Existing hot paths read ``token[0]`` and compare token identity.  A plain
    list also allowed callers to rewind that value and make stale warm plans
    appear current.  This private list subtype preserves those reads while
    accepting only the single legitimate transition ``current -> current+1``.
    Direct calls to ``list.__setitem__`` are interpreter-level bypasses and
    remain outside the supported mutation surface.
    """

    __slots__ = ()

    def __init__(self, value: int = 0) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("qualified mutation epoch must be nonnegative")
        list.__init__(self, [value])

    def __setitem__(self, index: Any, value: Any) -> None:
        if (
            type(index) is not int
            or index not in {0, -1}
            or type(value) is not int
            or value != list.__getitem__(self, 0) + 1
        ):
            raise ValueError(
                "qualified mutation epoch can only advance by one"
            )
        list.__setitem__(self, 0, value)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_QualifiedMutationEpoch":
        made = type(self)(list.__getitem__(self, 0))
        memo[id(self)] = made
        return made

    def __reduce_ex__(self, protocol: int) -> Tuple[Any, Tuple[int]]:
        del protocol
        return type(self), (list.__getitem__(self, 0),)

    def _reject_resize(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("qualified mutation epoch has fixed length")

    __delitem__ = _reject_resize
    __iadd__ = _reject_resize
    __imul__ = _reject_resize
    append = _reject_resize
    clear = _reject_resize
    extend = _reject_resize
    insert = _reject_resize
    pop = _reject_resize
    remove = _reject_resize
    reverse = _reject_resize
    sort = _reject_resize


def _bind_qualified_direct_state_token(value: Any, token: list[int]) -> None:
    """Subscribe shared model objects to every owning mesh mutation epoch."""

    subscriptions = value.__dict__.get("_qualified_direct_state_tokens")
    if subscriptions is None:
        subscriptions = []
        object.__setattr__(value, "_qualified_direct_state_tokens", subscriptions)
    if not any(bound is token for bound in subscriptions):
        subscriptions.append(token)
    # Preserve the singular attribute as the most recent owner for concise
    # diagnostics and backward-compatible introspection.
    object.__setattr__(value, "_qualified_direct_state_token", token)


class _QualifiedStateMapping(dict):
    """A public-dict-compatible mapping that advances one shared epoch."""

    def __init__(
        self,
        values: Any = (),
        token: Optional[list[int]] = None,
        kind: str = "detached",
    ) -> None:
        dict.__init__(self)
        self._qualified_token = (
            _QualifiedMutationEpoch() if token is None else token
        )
        self._qualified_kind = str(kind)
        entries = values.items() if isinstance(values, Mapping) else values
        for key, value in entries:
            if token is None:
                dict.__setitem__(self, key, value)
                continue
            self._prepare(value)
            dict.__setitem__(self, key, value)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_QualifiedStateMapping":
        token = copy.deepcopy(self._qualified_token, memo)
        values = {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self.items()
        }
        made = type(self)(values, token, self._qualified_kind)
        memo[id(self)] = made
        return made

    def __reduce_ex__(self, protocol: int) -> Tuple[Any, Tuple[Any, ...]]:
        del protocol
        return (
            type(self),
            (dict(self), self._qualified_token, self._qualified_kind),
        )

    def _prepare(self, value: Any) -> None:
        if self._qualified_kind == "detached":
            return
        if self._qualified_kind == "element":
            _freeze_qualified_element_vector_inputs(value)
        _bind_qualified_direct_state_token(value, self._qualified_token)

    def _advance(self) -> None:
        self._qualified_token[0] = int(self._qualified_token[0]) + 1

    def __setitem__(self, key: Any, value: Any) -> None:
        self._prepare(value)
        dict.__setitem__(self, key, value)
        self._advance()

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, key)
        self._advance()

    def clear(self) -> None:
        if self:
            dict.clear(self)
            self._advance()

    def pop(self, key: Any, *default: Any) -> Any:
        existed = key in self
        value = dict.pop(self, key, *default)
        if existed:
            self._advance()
        return value

    def popitem(self) -> Tuple[Any, Any]:
        value = dict.popitem(self)
        self._advance()
        return value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return dict.__getitem__(self, key)
        self[key] = default
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value

    def __ior__(self, other: Any) -> "_QualifiedStateMapping":
        self.update(other)
        return self


def _ensure_qualified_state_mappings(mesh: "FEMesh") -> None:
    """Install tracked public mappings, including after wholesale replacement."""

    token = mesh._qualified_direct_state_token
    for name, kind in (("nodes", "node"), ("elements", "element")):
        current = getattr(mesh, name)
        if (
            isinstance(current, _QualifiedStateMapping)
            and current._qualified_token is token
            and current._qualified_kind == kind
        ):
            continue
        object.__setattr__(
            mesh,
            name,
            _QualifiedStateMapping(dict(current), token, kind),
        )


class DOFManager:
    """
    Manages degrees of freedom for the FE model.

    Each node has 6 DOFs: [ux, uy, uz, rx, ry, rz]
    (displacements in x,y,z and rotations about x,y,z)
    """

    DOF_NAMES = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
    DOF_PER_NODE = 6

    def __init__(self):
        self._node_to_dof: Dict[int, List[int]] = {}
        self._dof_to_node: Dict[int, int] = {}
        self._dof_to_local: Dict[int, int] = {}
        self._total_dofs = 0
        self._constrained_dofs: set = set()

    def add_node(self, node_id: int) -> List[int]:
        """Add a new node and return its DOF indices."""
        if node_id in self._node_to_dof:
            raise ValueError(f"Node {node_id} already has assigned DOFs")
        dofs = list(range(self._total_dofs, self._total_dofs + self.DOF_PER_NODE))
        self._node_to_dof[node_id] = dofs
        for i, dof in enumerate(dofs):
            self._dof_to_node[dof] = node_id
            self._dof_to_local[dof] = i
        self._total_dofs += self.DOF_PER_NODE
        return dofs

    def get_node_dofs(self, node_id: int) -> List[int]:
        """Get all DOF indices for a node."""
        return self._node_to_dof.get(node_id, [])

    def get_dof_info(self, dof: int) -> Tuple[int, int, str]:
        """Get node ID, local DOF index, and name for a global DOF."""
        node_id = self._dof_to_node.get(dof, -1)
        local_idx = self._dof_to_local.get(dof, -1)
        name = self.DOF_NAMES[local_idx] if 0 <= local_idx < 6 else "unknown"
        return node_id, local_idx, name

    @property
    def total_dofs(self) -> int:
        return self._total_dofs

    @property
    def active_dofs(self) -> int:
        return self._total_dofs - len(self._constrained_dofs)

    def constrain_dof(self, dof: int):
        """Mark a DOF as constrained."""
        self._constrained_dofs.add(dof)

    def is_constrained(self, dof: int) -> bool:
        return dof in self._constrained_dofs

    def get_free_dofs(self) -> List[int]:
        """Get list of free (unconstrained) DOFs."""
        return [dof for dof in range(self._total_dofs) if dof not in self._constrained_dofs]

    def create_dof_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create arrays for constrained and free DOFs."""
        free_dofs = np.array(self.get_free_dofs(), dtype=int)
        constrained_dofs = np.array(list(self._constrained_dofs), dtype=int)
        return free_dofs, constrained_dofs


@dataclass
class Node:
    """A node in the FE mesh."""
    id: int
    x: float
    y: float
    z: float
    dofs: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep direct coordinate assignments observable by narrow cache plans
        # without changing the serialized dataclass fields.  Supported mesh
        # mutations still advance the mesh-wide geometry revision as before.
        object.__setattr__(self, "_coordinate_revision", 0)

    def __setattr__(self, name: str, value: Any) -> None:
        revision = self.__dict__.get("_coordinate_revision")
        if revision is not None and name in {"x", "y", "z"}:
            object.__setattr__(self, "_coordinate_revision", int(revision) + 1)
            tokens = self.__dict__.get("_qualified_direct_state_tokens")
            if tokens is None:
                token = self.__dict__.get("_qualified_direct_state_token")
                tokens = () if token is None else (token,)
            for token in tokens:
                token[0] = int(token[0]) + 1
        super().__setattr__(name, value)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Node":
        made = type(self).__new__(type(self))
        memo[id(self)] = made
        for name, value in self.__dict__.items():
            if name in {
                "_qualified_direct_state_token",
                "_qualified_direct_state_tokens",
            }:
                continue
            object.__setattr__(made, name, copy.deepcopy(value, memo))
        return made

    def coords(self) -> np.ndarray:
        """Return node coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])


@dataclass
class FEMesh:
    """
    Finite Element Mesh

    Stores nodes, elements, and connectivity for the FE model.
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, 'Element'] = field(default_factory=dict)
    dof_manager: DOFManager = field(default_factory=DOFManager)
    point_masses: Dict[int, float] = field(default_factory=dict)
    concentrated_masses: dict = field(default_factory=dict, kw_only=True)
    element_activity: Optional[object] = None
    revisions: Dict[str, int] = field(default_factory=lambda: {
        "topology": 0,
        "geometry": 0,
        "material": 0,
        "mass": 0,
        "load": 0,
        "boundary": 0,
        "mpc": 0,
        "result_state": 0,
    })

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep wholesale public state-mapping replacements observable."""

        token = self.__dict__.get("_qualified_direct_state_token")
        kind = {"nodes": "node", "elements": "element"}.get(name)
        if token is not None and kind is not None:
            already_tracked = (
                isinstance(value, _QualifiedStateMapping)
                and value._qualified_token is token
                and value._qualified_kind == kind
            )
            if not already_tracked:
                value = _QualifiedStateMapping(dict(value), token, kind)
                token[0] = int(token[0]) + 1
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        token = _QualifiedMutationEpoch()
        object.__setattr__(self, "_qualified_direct_state_token", token)
        _ensure_qualified_state_mappings(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FEMesh":
        """Copy owned model state while intentionally dropping derived caches."""

        made = type(self).__new__(type(self))
        memo[id(self)] = made
        for definition in fields(self):
            object.__setattr__(
                made,
                definition.name,
                copy.deepcopy(getattr(self, definition.name), memo),
            )
        for element in made.elements.values():
            _clear_element_local_caches(element)
            # NumPy deepcopy normally turns arrays backed by immutable bytes
            # into writable owners.  Qualified element deepcopy and this
            # mesh-level backstop both restore the cache-input invariant.
            _freeze_qualified_element_vector_inputs(element)
        for value in tuple(made.nodes.values()) + tuple(made.elements.values()):
            value.__dict__.pop("_qualified_direct_state_token", None)
            value.__dict__.pop("_qualified_direct_state_tokens", None)
        made.__post_init__()
        return made

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude bounded derived plans from Python serialization."""

        state = dict(self.__dict__)
        state.pop("_qualified_s3_reference_stiffness_plan", None)
        state.pop("_recovery_batch_plan", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        # Meshes pickled before concentrated masses existed lack this field.
        self.__dict__.setdefault("concentrated_masses", {})
        if "_qualified_direct_state_token" not in self.__dict__:
            object.__setattr__(
                self,
                "_qualified_direct_state_token",
                _QualifiedMutationEpoch(),
            )
        _ensure_qualified_state_mappings(self)

    def _advance_revision(self, category: str) -> None:
        """Increment one revision without applying invalidation policy."""

        self.revisions[category] = int(self.revisions.get(category, 0)) + 1

    def bump_revision(self, category: str) -> None:
        """Increment a mesh/model revision category and clear stale caches."""
        self._advance_revision(category)
        # Element-local matrices depend on an element's geometry and material,
        # not on unrelated elements or MPC topology.  Scanning every existing
        # element for every add_element() made model construction O(E**2).
        if category in {"geometry", "material"}:
            for element in self.elements.values():
                _clear_element_local_caches(element)
        if category in {"topology", "mpc"} and hasattr(self, "_sparsity_cache"):
            self._sparsity_cache = {}
        if category in {"topology", "mpc"} and hasattr(self, "_topology_signature_cache"):
            self._topology_signature_cache = {}

    def revision_signature(self) -> Dict[str, int]:
        signature = {
            key: int(value) for key, value in sorted(self.revisions.items())
        }
        if self.element_activity is not None:
            signature["activity"] = int(
                getattr(self.element_activity, "sequence", 0)
            )
        return signature

    def add_node(self, node_id: int, x: float, y: float, z: float) -> Node:
        """Add a node to the mesh."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        node = Node(id=node_id, x=x, y=y, z=z)
        _bind_qualified_direct_state_token(
            node, self._qualified_direct_state_token
        )
        node.dofs = self.dof_manager.add_node(node_id)
        self.nodes[node_id] = node
        self.bump_revision("topology")
        # A genuinely new node cannot be referenced by any already-valid
        # element, so existing element-local geometry caches remain valid.
        # The geometry revision still advances to invalidate model-wide plans.
        self._advance_revision("geometry")
        return node

    def add_element(self, element_id: int, element: 'Element'):
        """Add an element to the mesh."""
        # Elements can be precomputed before insertion (including against
        # another mesh).  Clear only the incoming element; revisiting every
        # existing element would turn construction into O(E**2).
        _clear_element_local_caches(element)
        _freeze_qualified_element_vector_inputs(element)
        _bind_qualified_direct_state_token(
            element, self._qualified_direct_state_token
        )
        self.elements[element_id] = element
        self.bump_revision("topology")
        self.bump_revision("mpc")

    def set_node_coordinates(self, node_id: int, x: float, y: float, z: float) -> None:
        """Update node coordinates and invalidate geometry-dependent caches."""
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        node.x = float(x)
        node.y = float(y)
        node.z = float(z)
        self.bump_revision("geometry")

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_element(self, element_id: int) -> Optional['Element']:
        return self.elements.get(element_id)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    def get_connectivity(self) -> Dict[int, List[int]]:
        """Get element connectivity (element_id -> node_ids)."""
        return {eid: elem.node_ids for eid, elem in self.elements.items()}

    def get_node_coordinates(self) -> np.ndarray:
        """Get coordinates of all nodes as array (n_nodes, 3)."""
        coords = np.zeros((self.num_nodes, 3))
        for i, (_node_id, node) in enumerate(self.nodes.items()):
            coords[i] = node.coords()
        return coords


@dataclass
class FEModel:
    """
    Complete Finite Element Model

    Contains mesh, materials, boundary conditions, loads, and results.
    """
    name: str
    mesh: FEMesh = field(default_factory=FEMesh)
    materials: Dict[str, StructuralMaterial] = field(default_factory=dict)
    boundary_conditions: List['BoundaryCondition'] = field(default_factory=list)
    load_cases: List['LoadCase'] = field(default_factory=list)
    current_material: str = "default"
    constraint_equations: List['ConstraintEquation'] = field(default_factory=list)

    def __post_init__(self):
        if "default" not in self.materials:
            self.materials["default"] = Material(
                name="default",
                elastic_modulus=210e9,  # Steel
                poisson_ratio=0.3
            )

    def set_element_activity(self, activity: Optional[object]) -> Optional[object]:
        """Attach activity state for exactly the mesh's stable element IDs."""

        if activity is not None:
            managed = {
                int(element_id)
                for element_id in np.asarray(
                    getattr(activity, "element_ids", ()), dtype=np.int64
                ).reshape(-1)
            }
            actual = {int(element_id) for element_id in self.mesh.elements}
            if managed != actual:
                missing = sorted(actual - managed)
                extra = sorted(managed - actual)
                raise ValueError(
                    "ElementActivity IDs must exactly match the FE mesh; "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )
        self.mesh.element_activity = activity
        self.mesh.bump_revision("result_state")
        return activity

    def add_material(self, name: str, elastic_modulus: float, poisson_ratio: float,
                    density: float = 0.0, yield_stress: float = 0.0,
                    hardening_curve: Optional[object] = None) -> Material:
        """Add a material to the model."""
        mat = Material(
            name=name,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            density=density,
            yield_stress=yield_stress,
            hardening_curve=hardening_curve
        )
        self.materials[name] = mat
        self.mesh.bump_revision("material")
        return mat

    def register_material(self, material: StructuralMaterial) -> StructuralMaterial:
        """Register a solver-compatible material object.

        Registration is structural rather than inheritance-based: a future
        ANYmaterial object can satisfy :class:`StructuralMaterial` directly.
        A same-name registration replaces the previous material, matching
        :meth:`add_material`, and invalidates material-dependent caches.
        """

        validate_material(material)
        self.materials[material.name] = material
        self.mesh.bump_revision("material")
        return material

    def add_orthotropic_material(
        self,
        name: str,
        elastic_modulus_1: float,
        elastic_modulus_2: float,
        elastic_modulus_3: float,
        poisson_ratio_12: float,
        poisson_ratio_13: float,
        poisson_ratio_23: float,
        shear_modulus_12: float,
        shear_modulus_13: float,
        shear_modulus_23: float,
        density: float = 0.0,
        hill_yield: Optional[Hill48Yield] = None,
        hardening_curve: Optional[object] = None,
    ) -> OrthotropicMaterial:
        """Construct and register a homogeneous orthotropic material."""

        material = OrthotropicMaterial(
            name=name,
            elastic_modulus_1=elastic_modulus_1,
            elastic_modulus_2=elastic_modulus_2,
            elastic_modulus_3=elastic_modulus_3,
            poisson_ratio_12=poisson_ratio_12,
            poisson_ratio_13=poisson_ratio_13,
            poisson_ratio_23=poisson_ratio_23,
            shear_modulus_12=shear_modulus_12,
            shear_modulus_13=shear_modulus_13,
            shear_modulus_23=shear_modulus_23,
            density=density,
            hill_yield=hill_yield,
            hardening_curve=hardening_curve,
        )
        self.register_material(material)
        return material

    def set_material(self, name: str):
        """Set the current material for new elements."""
        if name not in self.materials:
            raise ValueError(f"Material '{name}' not found")
        self.current_material = name

    def get_material(self, name: str = None) -> StructuralMaterial:
        """Get a material by name, or the current material."""
        name = name or self.current_material
        return self.materials.get(name, self.materials["default"])

    def add_node(self, node_id: int, x: float, y: float, z: float) -> Node:
        """Add a node to the model."""
        return self.mesh.add_node(node_id, x, y, z)

    def add_element(self, element_id: int, element: 'Element'):
        """Add an element to the model."""
        self.mesh.add_element(element_id, element)

    def add_point_mass(self, node_id: int, mass: float):
        """Attach a lumped translational point mass to a node.

        The mass enters the global mass matrix (so it shifts natural
        frequencies and participates in transient/collision dynamics) and, when
        an acceleration/gravity field is applied, produces the corresponding
        inertial load.
        """
        node_id = int(node_id)
        if self.mesh.get_node(node_id) is None:
            raise ValueError(f"Cannot attach point mass to missing node {node_id}")
        mass = float(mass)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError("Point mass must be finite and non-negative")
        if mass == 0.0:
            return
        self.mesh.point_masses[node_id] = self.mesh.point_masses.get(node_id, 0.0) + mass
        self.mesh.bump_revision("mass")

    def add_concentrated_mass(self, node_id, mass, *, offset=(0.,0.,0.), inertia_cg=((0.,0.,0.),)*3):
        """Add physical mass and CG inertia, rigidly attached at a six-DOF node."""
        from .concentrated_mass import ConcentratedMass
        node = self.mesh.get_node(int(node_id))
        if node is None or len(node.dofs) != 6:
            raise ValueError("concentrated mass requires an existing six-DOF attachment node")
        value = ConcentratedMass(mass, offset, inertia_cg)
        self.mesh.concentrated_masses.setdefault(int(node_id), []).append(value)
        self.mesh.bump_revision("mass")

    def add_boundary_condition(self, bc: 'BoundaryCondition'):
        """Add a boundary condition to the model."""
        self.boundary_conditions.append(bc)
        self.mesh.bump_revision("boundary")

    def add_constraint_equation(
        self,
        equation: Optional['ConstraintEquation'] = None,
        *,
        terms: Optional[Tuple[Tuple[int, float], ...]] = None,
        rhs: float = 0.0,
        source_id: str = "",
        dependent_dof: Optional[int] = None,
    ) -> 'ConstraintEquation':
        """Add a generalized affine constraint to the common reduction path.

        Callers may pass an existing :class:`ConstraintEquation` or its public
        construction fields.  The first term is the pivot when
        ``dependent_dof`` is omitted.
        """

        from .constraint_audit import ConstraintEquation

        if equation is not None and terms is not None:
            raise ValueError("Pass either equation or terms, not both")
        if equation is None:
            if terms is None:
                raise ValueError("terms are required")
            equation = ConstraintEquation(
                terms=terms,
                rhs=rhs,
                source_id=source_id,
                dependent_dof=dependent_dof,
            )
        if not isinstance(equation, ConstraintEquation):
            raise TypeError("equation must be a ConstraintEquation")
        self.constraint_equations.append(equation)
        self.mesh.bump_revision("boundary")
        return equation

    def add_load_case(self, load_case: 'LoadCase'):
        """Add a load case to the model."""
        self.load_cases.append(load_case)
        self.mesh.bump_revision("load")

    def apply_boundary_conditions(self):
        """Apply all boundary conditions to the mesh DOF manager."""
        # Keep invalid supports/MPCs from reaching element assembly or sparse
        # factorization in every analysis family.
        from .constraint_audit import require_valid_constraints

        require_valid_constraints(self)
        for bc in self.boundary_conditions:
            bc.apply(self.mesh.dof_manager)

    def clear_boundary_conditions(self):
        """Clear all boundary conditions."""
        self.boundary_conditions.clear()
        self.constraint_equations.clear()
        self.mesh.dof_manager = DOFManager()
        # Re-add nodes to reset DOFs
        for node_id, node in self.mesh.nodes.items():
            node.dofs = self.mesh.dof_manager.add_node(node_id)
        self.mesh.bump_revision("boundary")
        self.mesh.bump_revision("topology")

    def set_node_coordinates(self, node_id: int, x: float, y: float, z: float) -> None:
        """Update node coordinates and invalidate geometry-dependent caches."""
        self.mesh.set_node_coordinates(node_id, x, y, z)

    def bump_revision(self, category: str) -> None:
        self.mesh.bump_revision(category)

    def revision_signature(self) -> Dict[str, int]:
        return self.mesh.revision_signature()
