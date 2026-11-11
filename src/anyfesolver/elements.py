"""
Finite Element Implementations

This module contains element formulations for:
- ShellElement: 3/6-node triangular and 4/8-node quadrilateral Mindlin-Reissner shell element
- BeamElement: 2-node Timoshenko beam element with 6 DOF/node
- QuadraticBeamElement: 3-node quadratic Timoshenko beam element
- CoupledBeamShellElement: kinematic MPC element for eccentric beam-shell interaction

Shell convention
----------------
The shell element forms stiffness and stresses in an element-local orthonormal
basis at each integration point:

    local x = projected xi tangent direction
    local y = in-surface direction perpendicular to local x
    local z = shell normal

Global nodal translations and rotations are transformed to this local basis
before the membrane, bending and transverse shear B-matrices are evaluated.

Shell shear treatment
---------------------
Membrane and bending always use full integration for the element topology.
Transverse shear depends on the topology:

    * 4-node: MITC4 assumed natural shear (covariant shear sampled at the four
      edge midpoints and interpolated), integrated at the full 2x2 rule.  This
      avoids both shear locking and the spurious zero-energy w-hourglass mode
      of one-point reduced shear integration.
    * 3-node: centroidal edge-compatible assumed shear.  This is the constant
      shear part of the DSG/MITC3 family: transverse shear is evaluated from
      the element-average shear gap in the centroid frame, preserving rigid
      body motion and constant-shear patches without the locking-prone fully
      integrated linear Mindlin shear term.
    * 6-node: quadratic displacement interpolation with reduced three-point
      triangular shear integration.
    * 8-node: reduced 2x2 shear integration (S8R style).  When full element
      reduced integration is requested, a small nullspace-projection
      hourglass stiffness stabilizes modes outside the six rigid-body modes.

Beam section convention
-----------------------
Beam local axes are (x = member axis, y, z).  ``Iy`` is the second moment of
area about the local y axis and governs bending that deflects the beam in
local z; ``Iz`` governs deflection in local y.  ``shear_factor_y`` scales the
shear area for transverse shear force in local y, ``shear_factor_z`` for local
z.  The optional ``cross_section["orientation"]`` vector prescribes the local
z direction (e.g. the stiffener web direction).  Without it, a heuristic picks
local y close to global Y (or global Z for members nearly parallel to Y),
which leaves the section orientation unconstrained for asymmetric sections.

Beam-shell coupling
-------------------
Beam-shell coupling is represented as a linear multi-point constraint (MPC),
not as a large penalty spring.  For a beam node offset from a shell node by
vector r, the coupling relation is:

    u_beam = u_shell + theta_shell x r
    theta_beam = theta_shell

The assembly solver eliminates these slave beam DOFs through a transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from types import FunctionType, MemberDescriptorType
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from anymaterial import IsotropicMaterial as _RecoveryIsotropicMaterial
from . import _compensated_contractions as _local_contractions

from .beam_sections import (
    GENERALIZED_BEAM_RESULTANT_ORDER,
    GENERALIZED_BEAM_STRAIN_ORDER,
    generalized_beam_mass_matrix,
    generalized_beam_stiffness,
    resolve_generalized_beam_section,
)
from .jit_compiler import njit
from .material_curves import FiberSectionPlasticityConfig
from .materials import (
    beam_material_properties as _canonical_beam_material_properties,
    elastic_compliance_matrix as _canonical_elastic_compliance,
    material_symmetry as _canonical_material_symmetry,
)
from .plasticity import lobatto_layers, plane_stress_elastic_matrix, plane_stress_return_map
from .shell_sections import (
    SHELL_MEMBRANE_VOIGT_ORDER,
    SHELL_TRANSVERSE_SHEAR_ORDER,
    GeneralizedShellSection,
    coerce_generalized_shell_section,
)

if TYPE_CHECKING:
    from .fe_core import FEMesh, Material


_SMALL = 1.0e-12
_INITIAL_SHELL_STATE_KEYS = (
    "initial_membrane_stress",
    "initial_bending_stress",
    "initial_membrane_prestrain",
    "initial_curvature_prestrain",
    "initial_field_provenance",
)
_INITIAL_BEAM_STATE_KEYS = (
    "initial_fiber_stress",
    "initial_fiber_prestrain",
    "initial_field_provenance",
)


_OBSERVATION_MISSING = object()
_NO_ATTRIBUTE_DEFAULT = object()


def _run_post_observation_guard(
    post_observation: Optional[Callable[[], None]],
) -> None:
    """Recheck a caller-supplied exact authority lease after one observation."""

    if post_observation is not None:
        post_observation()


def _guarded_observe_attribute(
    owner: Any,
    name: str,
    *,
    default: Any = _NO_ATTRIBUTE_DEFAULT,
    post_observation: Optional[Callable[[], None]] = None,
) -> Any:
    """Read one external attribute and recheck before consuming its value."""

    if default is _NO_ATTRIBUTE_DEFAULT:
        value = getattr(owner, name)
    else:
        value = getattr(owner, name, default)
    _run_post_observation_guard(post_observation)
    return value


def _guarded_observe_call(
    provider: Any,
    *args: Any,
    post_observation: Optional[Callable[[], None]] = None,
    **kwargs: Any,
) -> Any:
    """Invoke one external provider and recheck before consuming its result."""

    value = provider(*args, **kwargs)
    _run_post_observation_guard(post_observation)
    return value


def _guarded_float(
    value: Any,
    *,
    post_observation: Optional[Callable[[], None]] = None,
) -> float:
    made = float(value)
    if type(value) is not float:
        _run_post_observation_guard(post_observation)
    return made


def _guarded_array(
    value: Any,
    *,
    dtype: Any = np.float64,
    post_observation: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    made = np.asarray(value, dtype=dtype)
    # Converting an exact numeric ndarray runs no foreign code.
    pure = type(value) is np.ndarray and value.dtype.kind in "fiub"
    if not pure:
        _run_post_observation_guard(post_observation)
    owned = np.array(made, dtype=dtype, order="C", copy=True)
    if not pure:
        _run_post_observation_guard(post_observation)
    return owned


def _guarded_owned_mapping(
    value: Any,
    *,
    label: str,
    post_observation: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Detach a string-keyed external mapping before later evaluation."""

    items_provider = _guarded_observe_attribute(
        value,
        "items",
        post_observation=post_observation,
    )
    observed_items = _guarded_observe_call(
        items_provider,
        post_observation=post_observation,
    )
    items = tuple(observed_items)
    _run_post_observation_guard(post_observation)
    owned: Dict[str, Any] = {}
    for item in items:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
            raise TypeError(f"{label} must contain exact string keys")
        key, member = item
        if key in owned:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        owned[key] = member
    return owned


def _guarded_owned_plain_value(
    value: Any,
    *,
    label: str,
    post_observation: Optional[Callable[[], None]] = None,
) -> Any:
    """Recursively detach caller-owned state before formulation mechanics."""

    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if type(value) is np.ndarray:
        return _guarded_array(
            value,
            dtype=value.dtype,
            post_observation=post_observation,
        )
    if isinstance(value, Mapping):
        observed = _guarded_owned_mapping(
            value,
            label=label,
            post_observation=post_observation,
        )
        return {
            key: _guarded_owned_plain_value(
                member,
                label=f"{label}.{key}",
                post_observation=post_observation,
            )
            for key, member in observed.items()
        }
    if type(value) in {list, tuple}:
        observed = tuple(value)
        _run_post_observation_guard(post_observation)
        made = [
            _guarded_owned_plain_value(
                member,
                label=f"{label}[{index}]",
                post_observation=post_observation,
            )
            for index, member in enumerate(observed)
        ]
        return made if type(value) is list else tuple(made)
    try:
        return _guarded_array(
            value,
            post_observation=post_observation,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} contains an unsupported external value") from exc


def _guarded_owned_generalized_shell_section(
    section: Any,
    *,
    post_observation: Optional[Callable[[], None]] = None,
    _section_class: type[GeneralizedShellSection] = GeneralizedShellSection,
) -> Optional[GeneralizedShellSection]:
    """Coerce one section protocol to an exact solver-owned section."""

    if section is None:
        return None
    if type(section) is _section_class:
        return section
    if isinstance(section, Mapping):
        source = _guarded_owned_mapping(
            section,
            label="generalized shell section",
            post_observation=post_observation,
        )

        def read(label: str, default: Any = None) -> Any:
            return source[label] if label in source else default

    else:

        def read(label: str, default: Any = None) -> Any:
            return _guarded_observe_attribute(
                section,
                label,
                default=default,
                post_observation=post_observation,
            )

    matrices: Dict[str, np.ndarray] = {}
    for name in ("A", "B", "D", "As"):
        raw = read(name, _OBSERVATION_MISSING)
        if raw is _OBSERVATION_MISSING or raw is None:
            raise TypeError(
                "Generalized shell section must provide A, B, D and As; "
                f"missing {name}"
            )
        matrices[name] = _guarded_array(
            raw,
            post_observation=post_observation,
        )
    raw_name = read("name", "")
    name = str(raw_name)
    _run_post_observation_guard(post_observation)
    masses: Dict[str, Optional[float]] = {}
    for field in ("mass_per_area", "rotary_inertia_per_area"):
        raw = read(field, None)
        masses[field] = (
            None
            if raw is None
            else _guarded_float(raw, post_observation=post_observation)
        )
    return _section_class(
        A=matrices["A"],
        B=matrices["B"],
        D=matrices["D"],
        As=matrices["As"],
        name=name,
        mass_per_area=masses["mass_per_area"],
        rotary_inertia_per_area=masses["rotary_inertia_per_area"],
    )


def _immutable_float64_array(value: Any) -> np.ndarray:
    """Return a C-contiguous float64 array backed by immutable bytes.

    Shell quadrature tables are process-wide formulation inputs.  A normal
    ``setflags(write=False)`` array can be made writable again when it owns its
    storage; a bytes-backed view cannot.  This prevents one element or test
    from silently changing the rule used by every other shell element.
    """

    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float64
    ).reshape(contiguous.shape)


def _copy_state_fields(state: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    copied: Dict[str, Any] = {}
    for key in keys:
        if key not in state:
            continue
        value = state[key]
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _shell_field_rows(value: Any, n_gp: int, name: str) -> np.ndarray:
    """Broadcast a shell-local engineering vector to one row per Gauss point."""
    array = np.asarray(value, dtype=float)
    if array.shape == (3,):
        return np.broadcast_to(array, (n_gp, 3)).copy()
    if array.shape == (1, 3):
        return np.broadcast_to(array, (n_gp, 3)).copy()
    if array.shape == (n_gp, 3):
        return array.copy()
    raise ValueError(f"{name} must have shape (3,) or ({n_gp}, 3); got {array.shape}")


def _beam_fiber_field(
    value: Any,
    total_fibers: int,
    section_fibers: int,
    name: str,
) -> np.ndarray:
    """Broadcast a beam-fiber field over one or all integration sections."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(total_fibers, float(array[0]), dtype=float)
    if array.size == section_fibers and total_fibers % section_fibers == 0:
        return np.tile(array, total_fibers // section_fibers)
    if array.size == total_fibers:
        return array.copy()
    raise ValueError(
        f"{name} must be scalar, have {section_fibers} section-fiber values, "
        f"or {total_fibers} total values; got {array.size}"
    )


@njit
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors without np.cross dispatch overhead."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
    )


@njit
def _inv2(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Inverse and determinant of a 2x2 matrix without LAPACK overhead."""
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < _SMALL:
        raise np.linalg.LinAlgError("singular 2x2 matrix")
    inv = np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
    ) / det
    return inv, float(det)


def _section_orientation(cross_section: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the prescribed local-z (web) direction from a cross-section dict."""
    value = cross_section.get("orientation", cross_section.get("web_direction"))
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size < 3 or float(np.linalg.norm(vector[:3])) < _SMALL:
        return None
    return np.array(vector[:3], dtype=float)


def _beam_rotation_matrix(e1: np.ndarray, orientation: Optional[np.ndarray]) -> np.ndarray:
    """Build the beam local frame [e1 e2 e3] with optional prescribed local z.

    ``orientation`` is the requested local z direction (section web direction).
    It is projected perpendicular to the member axis; if it is (nearly)
    parallel to the axis the heuristic fallback is used instead.
    """
    if orientation is not None:
        candidate = orientation - np.dot(orientation, e1) * e1
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-6 * float(np.linalg.norm(orientation)):
            e3 = candidate / norm
            e2 = _cross3(e3, e1)
            e2 /= np.linalg.norm(e2)
            return np.column_stack((e1, e2, e3))
    trial = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(e1, trial))) > 0.95:
        trial = np.array([0.0, 0.0, 1.0])
    e2 = trial - np.dot(trial, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = _cross3(e1, e2)
    e3 /= np.linalg.norm(e3)
    return np.column_stack((e1, e2, e3))


def _elastic_symmetry(
    material: Any,
    post_observation: Optional[Callable[[], None]] = None,
) -> str:
    """Return the normalized elastic symmetry without inventing isotropic aliases."""

    symmetry = _canonical_material_symmetry(material)
    _run_post_observation_guard(post_observation)
    if symmetry not in {"isotropic", "orthotropic"}:
        raise ValueError(
            f"Unsupported elastic symmetry {symmetry!r}; ANYfeSolver currently supports "
            "only isotropic and orthotropic materials."
        )
    return symmetry


def _elastic_compliance(
    material: Any,
    post_observation: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Engineering compliance in [11,22,33,23,13,12] order."""

    if post_observation is None:
        raw_compliance = _canonical_elastic_compliance(material)
    else:
        provider = _guarded_observe_attribute(
            material,
            "elastic_compliance_matrix",
            default=None,
            post_observation=post_observation,
        )
        if callable(provider):
            raw_compliance = _guarded_observe_call(
                provider,
                post_observation=post_observation,
            )
        elif _elastic_symmetry(material, post_observation) == "isotropic":
            elastic_modulus = _guarded_float(
                _guarded_observe_attribute(
                    material,
                    "elastic_modulus",
                    post_observation=post_observation,
                ),
                post_observation=post_observation,
            )
            poisson_ratio = _guarded_float(
                _guarded_observe_attribute(
                    material,
                    "poisson_ratio",
                    post_observation=post_observation,
                ),
                post_observation=post_observation,
            )
            shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
            raw_compliance = np.array(
                [
                    [
                        1.0 / elastic_modulus,
                        -poisson_ratio / elastic_modulus,
                        -poisson_ratio / elastic_modulus,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        -poisson_ratio / elastic_modulus,
                        1.0 / elastic_modulus,
                        -poisson_ratio / elastic_modulus,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        -poisson_ratio / elastic_modulus,
                        -poisson_ratio / elastic_modulus,
                        1.0 / elastic_modulus,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0 / shear_modulus, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0 / shear_modulus, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / shear_modulus],
                ],
                dtype=float,
            )
        else:
            raise ValueError(
                "Structural material must provide elastic_compliance_matrix()"
            )
    compliance = np.asarray(raw_compliance, dtype=float)
    _run_post_observation_guard(post_observation)
    if compliance.shape != (6, 6) or not np.all(np.isfinite(compliance)):
        raise ValueError("elastic_compliance_matrix() must return a finite 6x6 matrix")
    if not np.allclose(compliance, compliance.T, rtol=1.0e-10, atol=1.0e-18):
        raise ValueError("elastic compliance matrix must be symmetric")
    return 0.5 * (compliance + compliance.T)


def _shell_elastic_material_cache_fingerprint(
    material: Any,
    post_observation: Optional[Callable[[], None]] = None,
) -> Tuple[Any, ...]:
    """Bind every elastic input consumed by ``_shell_material_matrices``.

    Material objects are intentionally public and may be edited in place.  A
    cache key based only on object identity therefore admits stale shell
    mechanics.  Keep the isotropic branch bit-for-bit aligned with the actual
    scalar path, including an explicitly supplied shear modulus; use the
    canonical compliance bytes for every other supported material protocol.
    """

    symmetry = _elastic_symmetry(material, post_observation)
    raw_elastic_modulus = _guarded_observe_attribute(
        material,
        "elastic_modulus",
        default=_OBSERVATION_MISSING,
        post_observation=post_observation,
    )
    raw_poisson_ratio = _guarded_observe_attribute(
        material,
        "poisson_ratio",
        default=_OBSERVATION_MISSING,
        post_observation=post_observation,
    )
    if (
        symmetry == "isotropic"
        and raw_elastic_modulus is not _OBSERVATION_MISSING
        and raw_poisson_ratio is not _OBSERVATION_MISSING
    ):
        elastic_modulus = _guarded_float(
            raw_elastic_modulus,
            post_observation=post_observation,
        )
        poisson_ratio = _guarded_float(
            raw_poisson_ratio,
            post_observation=post_observation,
        )
        raw_shear_modulus = _guarded_observe_attribute(
            material,
            "shear_modulus",
            default=elastic_modulus / (2.0 * (1.0 + poisson_ratio)),
            post_observation=post_observation,
        )
        shear_modulus = _guarded_float(
            raw_shear_modulus,
            post_observation=post_observation,
        )
        return (
            "isotropic_scalar_path",
            elastic_modulus,
            poisson_ratio,
            shear_modulus,
        )
    compliance = np.ascontiguousarray(
        _elastic_compliance(material, post_observation),
        dtype=np.float64,
    )
    _run_post_observation_guard(post_observation)
    return (
        symmetry,
        compliance.shape,
        compliance.tobytes(order="C"),
    )


def _generalized_shell_section_cache_fingerprint(
    section: Any,
    post_observation: Optional[Callable[[], None]] = None,
) -> Optional[Tuple[Any, ...]]:
    """Return the complete stiffness/mass identity of a generalized section."""

    if section is None:
        return None
    matrices = []
    for name, shape in (("A", (3, 3)), ("B", (3, 3)), ("D", (3, 3)), ("As", (2, 2))):
        raw = _guarded_observe_attribute(
            section,
            name,
            post_observation=post_observation,
        )
        value = _guarded_array(
            raw,
            post_observation=post_observation,
        )
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"Generalized shell section {name} must remain a finite {shape} matrix"
            )
        matrices.append((name, shape, value.tobytes(order="C")))
    raw_name = _guarded_observe_attribute(
        section,
        "name",
        default="",
        post_observation=post_observation,
    )
    name = str(raw_name)
    _run_post_observation_guard(post_observation)
    mass_values = []
    for field in ("mass_per_area", "rotary_inertia_per_area"):
        raw = _guarded_observe_attribute(
            section,
            field,
            default=None,
            post_observation=post_observation,
        )
        mass_values.append(
            None
            if raw is None
            else _guarded_float(raw, post_observation=post_observation)
        )
    return (
        tuple(matrices),
        name,
        *mass_values,
    )


def _in_plane_transform_matrices(angle_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engineering-strain, stress, and transverse-shear transforms.

    The material 1-axis is rotated by ``angle_rad`` from element-local x
    toward local y about the positive shell normal.
    """

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    axes = np.array([[c, -s], [s, c]], dtype=float)
    strain_to_material = np.zeros((3, 3), dtype=float)
    stress_to_local = np.zeros((3, 3), dtype=float)
    for column in range(3):
        engineering_strain = np.zeros(3, dtype=float)
        engineering_strain[column] = 1.0
        local_tensor = np.array(
            [
                [engineering_strain[0], 0.5 * engineering_strain[2]],
                [0.5 * engineering_strain[2], engineering_strain[1]],
            ],
            dtype=float,
        )
        material_tensor = axes.T @ local_tensor @ axes
        strain_to_material[:, column] = (
            material_tensor[0, 0],
            material_tensor[1, 1],
            2.0 * material_tensor[0, 1],
        )
        material_stress = np.zeros(3, dtype=float)
        material_stress[column] = 1.0
        stress_tensor = np.array(
            [
                [material_stress[0], material_stress[2]],
                [material_stress[2], material_stress[1]],
            ],
            dtype=float,
        )
        local_stress = axes @ stress_tensor @ axes.T
        stress_to_local[:, column] = (
            local_stress[0, 0],
            local_stress[1, 1],
            local_stress[0, 1],
        )
    return strain_to_material, stress_to_local, axes


def _compensated_elastic_zero_tangent(element, mesh, material, post_observation=None):
    """Scalar zero-state realization, called only after qualified elastic admission.

    Preserve scalar coefficient operations, including reciprocal-compliance
    drill, independently of the batch coefficient construction. No state/cache
    or caller eligibility is changed here.
    """
    geometry = element._nonlinear_geometry(mesh)
    membrane, shear, _strain, _stress = _shell_material_matrices(
        material, element._material_angle(geometry["R0"]), post_observation)
    modulus = 1.0 / float(_elastic_compliance(material, post_observation)[5, 5])
    h = element.thickness
    result = _local_contractions.shell_zero_tangent(
        geometry, h * membrane, h**3 / 12.0 * membrane,
        shear * (5.0 / 6.0) * h, modulus * h * element.drilling_stabilization)
    if post_observation is not None:
        post_observation()
    return result


def _shell_material_matrices(
    material: Any,
    angle_rad: float,
    post_observation: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local plane-stress/shear matrices and material-axis transforms."""

    symmetry = _elastic_symmetry(material, post_observation)
    # Preserve the legacy isotropic arithmetic exactly.  Forming and reducing
    # the 3-D compliance is mathematically equivalent, but its final-bit
    # roundoff can perturb tightly qualified eigenvalue thresholds.  External
    # protocol objects without the historical scalar fields still use the
    # compliance path below.
    raw_elastic_modulus = _guarded_observe_attribute(
        material,
        "elastic_modulus",
        default=_OBSERVATION_MISSING,
        post_observation=post_observation,
    )
    raw_poisson_ratio = _guarded_observe_attribute(
        material,
        "poisson_ratio",
        default=_OBSERVATION_MISSING,
        post_observation=post_observation,
    )
    if (
        symmetry == "isotropic"
        and raw_elastic_modulus is not _OBSERVATION_MISSING
        and raw_poisson_ratio is not _OBSERVATION_MISSING
    ):
        E = _guarded_float(
            raw_elastic_modulus,
            post_observation=post_observation,
        )
        nu = _guarded_float(
            raw_poisson_ratio,
            post_observation=post_observation,
        )
        raw_shear_modulus = _guarded_observe_attribute(
            material,
            "shear_modulus",
            default=E / (2.0 * (1.0 + nu)),
            post_observation=post_observation,
        )
        G = _guarded_float(
            raw_shear_modulus,
            post_observation=post_observation,
        )
        Q_material = E / (1.0 - nu**2) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ],
            dtype=float,
        )
        return (
            Q_material,
            G * np.eye(2, dtype=float),
            np.eye(3, dtype=float),
            np.eye(3, dtype=float),
        )

    compliance = _elastic_compliance(material, post_observation)
    membrane_indices = np.array([0, 1, 5], dtype=np.intp)
    shear_indices = np.array([4, 3], dtype=np.intp)  # [13, 23]
    Q_material = np.linalg.inv(compliance[np.ix_(membrane_indices, membrane_indices)])
    G_material = np.linalg.inv(compliance[np.ix_(shear_indices, shear_indices)])
    strain_to_material, stress_to_local, axes = _in_plane_transform_matrices(angle_rad)
    Q_local = stress_to_local @ Q_material @ strain_to_material
    G_local = axes @ G_material @ axes.T
    return (
        0.5 * (Q_local + Q_local.T),
        0.5 * (G_local + G_local.T),
        strain_to_material,
        stress_to_local,
    )


def _beam_material_properties(material: Any, cross_section: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return E1, G12, G13 and torsional rigidity for a beam."""

    properties = _canonical_beam_material_properties(material)
    E1 = float(properties.axial_modulus)
    G12 = float(properties.shear_modulus_xy)
    G13 = float(properties.shear_modulus_xz)
    if _elastic_symmetry(material) == "orthotropic":
        value = cross_section.get("torsional_rigidity")
        try:
            torsional_rigidity = float(value)
        except (TypeError, ValueError):
            torsional_rigidity = float("nan")
        if not np.isfinite(torsional_rigidity) or torsional_rigidity <= 0.0:
            raise ValueError(
                "Orthotropic beam sections require a positive "
                "cross_section['torsional_rigidity'] in N*m^2."
            )
    else:
        torsional_rigidity = G12 * float(cross_section.get("J", 1.0e-8))
    return E1, G12, G13, torsional_rigidity


def _beam2_basic_deformation_matrix(length: float) -> np.ndarray:
    """Map local Beam2 DOFs to six work-conjugate basic deformations."""

    L = float(length)
    transform = np.zeros((6, 12), dtype=float)
    # Axial extension and torsional rotation.
    transform[0, 0], transform[0, 6] = -1.0, 1.0
    transform[1, 3], transform[1, 9] = -1.0, 1.0
    # Bending about y: rotations relative to the rigid chord rotation
    # ry + (w2-w1)/L.
    transform[2, 2], transform[2, 8], transform[2, 4] = (
        -1.0 / L,
        1.0 / L,
        1.0,
    )
    transform[3, 2], transform[3, 8], transform[3, 10] = (
        -1.0 / L,
        1.0 / L,
        1.0,
    )
    # Bending about z: rotations relative to the rigid chord rotation
    # rz - (v2-v1)/L.
    transform[4, 1], transform[4, 7], transform[4, 5] = (
        1.0 / L,
        -1.0 / L,
        1.0,
    )
    transform[5, 1], transform[5, 7], transform[5, 11] = (
        1.0 / L,
        -1.0 / L,
        1.0,
    )
    return transform


def _beam2_force_interpolation(length: float, xi: float) -> np.ndarray:
    """Map Beam2 basic forces to equilibrium section resultants."""

    L = float(length)
    x_over_length = 0.5 * (float(xi) + 1.0)
    interpolation = np.zeros((6, 6), dtype=float)
    interpolation[0, 0] = 1.0  # N
    interpolation[3, 1] = 1.0  # T
    # q[2:4] are the two basic local-y bending rotations.
    interpolation[2, 2:4] = 1.0 / L  # Vz
    interpolation[4, 2] = -1.0 + x_over_length  # My
    interpolation[4, 3] = x_over_length
    # q[4:6] are the two basic local-z bending rotations.  The sign of Vy
    # follows gamma_xy = v' - rz.
    interpolation[1, 4:6] = -1.0 / L  # Vy
    interpolation[5, 4] = -1.0 + x_over_length  # Mz
    interpolation[5, 5] = x_over_length
    return interpolation


def _beam2_generalized_flexibility(
    length: float,
    section_stiffness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact constant-section basic flexibility and stiffness."""

    L = float(length)
    compliance = np.linalg.inv(np.asarray(section_stiffness, dtype=float))
    flexibility = np.zeros((6, 6), dtype=float)
    points = (-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0))
    weights = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    for xi, weight in zip(points, weights):
        interpolation = _beam2_force_interpolation(L, xi)
        flexibility += (
            interpolation.T
            @ compliance
            @ interpolation
            * (L / 2.0 * weight)
        )
    return flexibility, np.linalg.inv(flexibility)


def _beam_generalized_strain_matrix_from_shape(
    shape_functions: np.ndarray,
    shape_derivatives: np.ndarray,
) -> np.ndarray:
    """Build a generalized beam B-matrix from compatible shape functions."""

    N = np.asarray(shape_functions, dtype=float).reshape(-1)
    derivative = np.asarray(shape_derivatives, dtype=float).reshape(-1)
    if derivative.size != N.size:
        raise ValueError("beam shape function and derivative sizes must match")
    B = np.zeros((6, 6 * N.size), dtype=float)
    for i in range(N.size):
        base = 6 * i
        B[0, base + 0] = derivative[i]  # eps_x
        B[1, base + 1] = derivative[i]  # gamma_xy = v' - rz
        B[1, base + 5] = -N[i]
        B[2, base + 2] = derivative[i]  # gamma_xz = w' + ry
        B[2, base + 4] = N[i]
        B[3, base + 3] = derivative[i]  # kappa_x
        B[4, base + 4] = derivative[i]  # kappa_y
        B[5, base + 5] = derivative[i]  # kappa_z
    return B


def _generalized_beam_recovery(
    strains: np.ndarray,
    resultants: np.ndarray,
    *,
    stations: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build resultants-only recovery without inventing physical stresses."""

    strain_values = np.asarray(strains, dtype=float)
    resultant_values = np.asarray(resultants, dtype=float)
    recovered: Dict[str, Any] = {
        "generalized_strain": strain_values.copy(),
        "generalized_resultant": resultant_values.copy(),
        "generalized_strain_order": GENERALIZED_BEAM_STRAIN_ORDER,
        "generalized_resultant_order": GENERALIZED_BEAM_RESULTANT_ORDER,
        "recovery_scope": "section_resultants_only",
        "physical_stress_available": False,
        "resultant_frame": "beam_local",
        "station_coordinate_system": "natural_xi",
    }
    for index, key in enumerate(GENERALIZED_BEAM_STRAIN_ORDER):
        recovered[key] = (
            float(strain_values[index])
            if strain_values.ndim == 1
            else strain_values[:, index].copy()
        )
    for index, key in enumerate(GENERALIZED_BEAM_RESULTANT_ORDER):
        recovered[key] = (
            float(resultant_values[index])
            if resultant_values.ndim == 1
            else resultant_values[:, index].copy()
        )
    descriptive_strains = (
        "axial_strain",
        "shear_strain_xy",
        "shear_strain_xz",
        "twist_rate",
        "curvature_y",
        "curvature_z",
    )
    descriptive_resultants = (
        "axial_force",
        "shear_force_y",
        "shear_force_z",
        "torsional_moment",
        "bending_moment_y",
        "bending_moment_z",
    )
    for index, key in enumerate(descriptive_strains):
        recovered[key] = (
            float(strain_values[index])
            if strain_values.ndim == 1
            else strain_values[:, index].copy()
        )
    for index, key in enumerate(descriptive_resultants):
        recovered[key] = (
            float(resultant_values[index])
            if resultant_values.ndim == 1
            else resultant_values[:, index].copy()
        )
    if stations is not None:
        recovered["station_coordinates"] = np.asarray(
            stations,
            dtype=float,
        ).copy()
    return recovered


def _beam_flow_scale(material: Any, curve: Any) -> float:
    """Scale a scalar hardening curve to the Hill longitudinal strength X.

    With no curve, the returned value is the constant perfect-plastic flow
    stress.  Otherwise it multiplies the curve so ``flow_stress(0) == X``.
    """

    if _elastic_symmetry(material) != "orthotropic":
        return 1.0
    hill_yield = getattr(material, "hill_yield", None)
    if hill_yield is None:
        raise ValueError(
            f"Orthotropic material {getattr(material, 'name', '<unnamed>')!r} "
            "requires hill_yield for beam fiber plasticity."
        )
    if curve is None:
        return float(hill_yield.X)
    reference = float(np.asarray(curve.flow_stress(np.array([0.0])), dtype=float)[0])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("hardening curve must provide a positive initial flow stress")
    return float(hill_yield.X) / reference


def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle rotation vector for a proper 3x3 rotation matrix."""
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace_value = float(np.trace(R))
    cos_angle = max(min((trace_value - 1.0) / 2.0, 1.0), -1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1.0e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        )
    if abs(np.pi - angle) < 1.0e-6:
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0) / 2.0)
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] *= -1.0
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] *= -1.0
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] *= -1.0
        norm = float(np.linalg.norm(axis))
        if norm <= _SMALL:
            return np.zeros(3, dtype=float)
        return angle * axis / norm
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return angle * axis


_STRICT_FLAT_S3_V2_CANDIDATE_ID = "CANDIDATE_E4_PL_S3_V2A_FLAT_LINEAR_V1"
_STRICT_FLAT_S3_V2B_CANDIDATE_ID = "CANDIDATE_E4_PL_S3_V2B_FLAT_LINEAR_V1"
_STRICT_FLAT_S3_V2C_CANDIDATE_ID = "CANDIDATE_E4_PL_S3_V2C_FLAT_LINEAR_PARITY_V1"
_S3_V2D_NATIVE_PARITY_CANDIDATE_ID = "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"


def _reject_strict_flat_s3_v2_legacy_dispatch(
    element: Any, operation: str
) -> None:
    if (
        getattr(type(element), "_legacy_shell_dispatch_forbidden", None)
        in {
            _STRICT_FLAT_S3_V2_CANDIDATE_ID,
            _STRICT_FLAT_S3_V2B_CANDIDATE_ID,
            _STRICT_FLAT_S3_V2C_CANDIDATE_ID,
            _S3_V2D_NATIVE_PARITY_CANDIDATE_ID,
        }
    ):
        from .element_capabilities import ElementCapabilityError

        raise ElementCapabilityError(
            "strict-flat S3 V2 forbids explicit legacy ShellElement dispatch: "
            + operation
        )


class Element(ABC):
    """Abstract base class for all FE elements."""

    def __init__(self, element_id: int, node_ids: List[int], material_name: str = "default"):
        self.element_id = element_id
        self.node_ids = node_ids
        self.material_name = material_name
        self._stiffness_matrix: Optional[np.ndarray] = None
        self._mass_matrix: Optional[np.ndarray] = None
        self._internal_forces: Optional[np.ndarray] = None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Element":
        """Copy instance state without asking ``copyreg`` about the class.

        The generic ``copy.deepcopy`` reconstruction path may lazily add a
        ``__slotnames__`` cache to a class.  Element class namespaces are part
        of the qualified Q4/S3 runtime authority, so even copying a legacy
        element must not mutate them.  Build the uninitialised instance
        directly, copy its instance dictionary, and explicitly copy any slots
        declared anywhere in a concrete subclass hierarchy.
        """

        _reject_strict_flat_s3_v2_legacy_dispatch(self, "Element.__deepcopy__")
        source_id = id(self)
        if source_id in memo:
            return memo[source_id]

        cls = type(self)
        made = object.__new__(cls)
        memo[source_id] = made

        try:
            source_namespace = object.__getattribute__(self, "__dict__")
            target_namespace = object.__getattribute__(made, "__dict__")
        except AttributeError:
            source_namespace = None
            target_namespace = None
        if source_namespace is not None and target_namespace is not None:
            for name, value in source_namespace.items():
                target_namespace[name] = copy.deepcopy(value, memo)

        for owner in type.__getattribute__(cls, "__mro__"):
            namespace = type.__getattribute__(owner, "__dict__")
            for slot_name, descriptor in namespace.items():
                if type(descriptor) is not MemberDescriptorType:
                    continue
                try:
                    value = object.__getattribute__(self, slot_name)
                except AttributeError:
                    continue
                object.__setattr__(made, slot_name, copy.deepcopy(value, memo))
        return made

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dofs_per_node(self) -> int:
        raise NotImplementedError

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    @abstractmethod
    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        raise NotImplementedError

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        _reject_strict_flat_s3_v2_legacy_dispatch(self, "Element.compute_mass_matrix")
        return np.zeros((self.total_dofs, self.total_dofs))

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        _reject_strict_flat_s3_v2_legacy_dispatch(
            self, "Element.compute_geometric_stiffness_matrix"
        )
        return np.zeros((self.total_dofs, self.total_dofs))

    def compute_internal_forces(
        self, mesh: "FEMesh", displacements: np.ndarray, material: "Material"
    ) -> np.ndarray:
        _reject_strict_flat_s3_v2_legacy_dispatch(
            self, "Element.compute_internal_forces"
        )
        return np.zeros(self.total_dofs)

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Internal force vector, tangent stiffness and trial state at u_elem.

        The default is linear elastic: F = K u with the constant stiffness as
        tangent.  Elements supporting geometric and/or material nonlinearity
        override this.  The returned state is a trial state; the incremental
        solver commits it only when the load step converges.  With
        ``tangent=False`` the stiffness entry may be None (used by the line
        search, which only needs residuals).
        """
        _reject_strict_flat_s3_v2_legacy_dispatch(
            self, "Element.compute_nonlinear_response"
        )
        K = self._stiffness_matrix
        if K is None:
            K = self.compute_stiffness_matrix(mesh, material)
        return K @ u_elem, (K if tangent else None), state

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, Any]:
        _reject_strict_flat_s3_v2_legacy_dispatch(self, "Element.compute_stresses")
        return {}

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        dof_mapping: List[int] = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node:
                dof_mapping.extend(node.dofs)
        return dof_mapping

    def _get_element_displacements(self, mesh: "FEMesh", displacements: np.ndarray) -> np.ndarray:
        u = np.asarray(displacements, dtype=float)
        if u.size == self.total_dofs:
            return u.copy()
        dof_mapping = self.get_dof_mapping(mesh)
        if not dof_mapping:
            return np.zeros(self.total_dofs)
        dof_indices = np.asarray(dof_mapping, dtype=np.intp)
        if int(dof_indices.max()) >= u.size:
            raise IndexError(
                f"Displacement vector has length {u.size}, but element {self.element_id} "
                f"requires global DOF {int(dof_indices.max())}."
            )
        return u[dof_indices]

    def to_dict(self) -> Dict[str, Any]:
        _reject_strict_flat_s3_v2_legacy_dispatch(self, "Element.to_dict")
        return {
            "element_id": self.element_id,
            "type": self.__class__.__name__,
            "node_ids": self.node_ids,
            "material_name": self.material_name,
        }


def _install_strict_flat_s3_v2_element_base_guards() -> None:
    """Capture the V2 guard in Element routes exposed by explicit dispatch."""

    from functools import wraps
    from inspect import signature

    guard = _reject_strict_flat_s3_v2_legacy_dispatch
    names = (
        "__deepcopy__",
        "compute_mass_matrix",
        "compute_geometric_stiffness_matrix",
        "compute_internal_forces",
        "compute_nonlinear_response",
        "compute_stresses",
        "to_dict",
    )
    for name in names:
        original = vars(Element)[name]

        @wraps(original)
        def guarded(
            self: Any,
            *args: Any,
            __name: str = name,
            __original: Callable[..., Any] = original,
            __guard: Callable[[Any, str], None] = guard,
            **kwargs: Any,
        ) -> Any:
            __guard(self, f"Element.{__name}")
            return __original(self, *args, **kwargs)

        guarded.__signature__ = signature(original)  # type: ignore[attr-defined]
        delattr(guarded, "__wrapped__")
        setattr(Element, name, guarded)


_install_strict_flat_s3_v2_element_base_guards()
del _install_strict_flat_s3_v2_element_base_guards


@njit
def _jit_compute_4node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
    )
    dN_dxi = np.array(
        [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ],
    )
    dN_deta = np.array(
        [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
    )
    return N, dN_dxi, dN_deta


@njit
def _jit_compute_8node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.zeros(8, dtype=float)
    N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta)
    N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta)
    N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta)
    N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta)
    N[4] = 0.5 * (1.0 - xi**2) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta**2)
    N[6] = 0.5 * (1.0 - xi**2) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta**2)

    dN_dxi = np.zeros(8, dtype=float)
    dN_dxi[0] = 0.25 * (1.0 - eta) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
    dN_dxi[1] = -0.25 * (1.0 - eta) * (1.0 - xi + eta) + 0.25 * (1.0 + xi) * (1.0 - eta)
    dN_dxi[2] = -0.25 * (1.0 + eta) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
    dN_dxi[3] = 0.25 * (1.0 + eta) * (1.0 + xi - eta) - 0.25 * (1.0 - xi) * (1.0 + eta)
    dN_dxi[4] = -xi * (1.0 - eta)
    dN_dxi[5] = 0.5 * (1.0 - eta**2)
    dN_dxi[6] = -xi * (1.0 + eta)
    dN_dxi[7] = -0.5 * (1.0 - eta**2)

    dN_deta = np.zeros(8, dtype=float)
    dN_deta[0] = 0.25 * (1.0 - xi) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
    dN_deta[1] = 0.25 * (1.0 + xi) * (1.0 - xi + eta) - 0.25 * (1.0 + xi) * (1.0 - eta)
    dN_deta[2] = -0.25 * (1.0 + xi) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
    dN_deta[3] = -0.25 * (1.0 - xi) * (1.0 + xi - eta) + 0.25 * (1.0 - xi) * (1.0 + eta)
    dN_deta[4] = -0.5 * (1.0 - xi**2)
    dN_deta[5] = -eta * (1.0 + xi)
    dN_deta[6] = 0.5 * (1.0 - xi**2)
    dN_deta[7] = -eta * (1.0 - xi)
    return N, dN_dxi, dN_deta


@njit
def _jit_compute_3node_shape_functions(r: float, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.array([1.0 - r - s, r, s])
    dN_dr = np.array([-1.0, 1.0, 0.0])
    dN_ds = np.array([-1.0, 0.0, 1.0])
    return N, dN_dr, dN_ds


@njit
def _jit_compute_6node_shape_functions(r: float, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1 = 1.0 - r - s
    l2 = r
    l3 = s

    N = np.zeros(6, dtype=float)
    N[0] = l1 * (2.0 * l1 - 1.0)
    N[1] = l2 * (2.0 * l2 - 1.0)
    N[2] = l3 * (2.0 * l3 - 1.0)
    N[3] = 4.0 * l1 * l2
    N[4] = 4.0 * l2 * l3
    N[5] = 4.0 * l3 * l1

    dN_dr = np.zeros(6, dtype=float)
    dN_dr[0] = 1.0 - 4.0 * l1
    dN_dr[1] = 4.0 * l2 - 1.0
    dN_dr[2] = 0.0
    dN_dr[3] = 4.0 * (l1 - l2)
    dN_dr[4] = 4.0 * l3
    dN_dr[5] = -4.0 * l3

    dN_ds = np.zeros(6, dtype=float)
    dN_ds[0] = 1.0 - 4.0 * l1
    dN_ds[1] = 0.0
    dN_ds[2] = 4.0 * l3 - 1.0
    dN_ds[3] = -4.0 * l2
    dN_ds[4] = 4.0 * l2
    dN_ds[5] = 4.0 * (l1 - l3)
    return N, dN_dr, dN_ds


@njit
def _jit_integrate_nonlinear_response(
    u_loc: np.ndarray,
    N_res: np.ndarray,
    M_res: np.ndarray,
    C0: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    B_m_all: np.ndarray,
    B_b_all: np.ndarray,
    B_d_all: np.ndarray,
    Gw_all: np.ndarray,
    detw_all: np.ndarray,
    B_s_all: np.ndarray,
    detw_shear_all: np.ndarray,
    D_shear: np.ndarray,
    drilling_stiffness: float,
    tangent: bool,
    has_plasticity: bool,
    n_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    F_loc = np.zeros(n_dof)
    K_loc = np.zeros((n_dof, n_dof))

    n_gp = len(detw_all)
    for g in range(n_gp):
        detw = detw_all[g]
        B_m = B_m_all[g]
        B_b = B_b_all[g]
        B_d = B_d_all[g]
        Gw = Gw_all[g]

        # Calculate theta = Gw @ u_loc
        theta_0 = 0.0
        theta_1 = 0.0
        for i in range(n_dof):
            theta_0 += Gw[0, i] * u_loc[i]
            theta_1 += Gw[1, i] * u_loc[i]

        # Calculate B_eff = B_m + B_nl
        B_eff = np.zeros((3, n_dof))
        for i in range(n_dof):
            B_eff[0, i] = B_m[0, i] + theta_0 * Gw[0, i]
            B_eff[1, i] = B_m[1, i] + theta_1 * Gw[1, i]
            B_eff[2, i] = B_m[2, i] + theta_0 * Gw[1, i] + theta_1 * Gw[0, i]

        # B_eff.T @ N_res[g] + B_b.T @ M_res[g]
        N_g = N_res[g]
        M_g = M_res[g]
        B_eff_T_N = np.zeros(n_dof)
        B_b_T_M = np.zeros(n_dof)
        for i in range(n_dof):
            B_eff_T_N[i] = B_eff[0, i] * N_g[0] + B_eff[1, i] * N_g[1] + B_eff[2, i] * N_g[2]
            B_b_T_M[i] = B_b[0, i] * M_g[0] + B_b[1, i] * M_g[1] + B_b[2, i] * M_g[2]

        # B_d @ u_loc
        Bd_u = 0.0
        for i in range(n_dof):
            Bd_u += B_d[0, i] * u_loc[i]

        for i in range(n_dof):
            F_loc[i] += (B_eff_T_N[i] + B_b_T_M[i]) * detw
            F_loc[i] += B_d[0, i] * (drilling_stiffness * Bd_u) * detw

        if not tangent:
            continue

        # K_loc += (B_eff.T @ C0[g] @ B_eff + B_b.T @ C2[g] @ B_b) * detw
        C0_g = C0[g]
        C2_g = C2[g]

        C0_B_eff = np.zeros((3, n_dof))
        C2_B_b = np.zeros((3, n_dof))
        for r in range(3):
            for c in range(n_dof):
                val0 = 0.0
                val2 = 0.0
                for k in range(3):
                    val0 += C0_g[r, k] * B_eff[k, c]
                    val2 += C2_g[r, k] * B_b[k, c]
                C0_B_eff[r, c] = val0
                C2_B_b[r, c] = val2

        for r in range(n_dof):
            for c in range(n_dof):
                val_eff = 0.0
                val_b = 0.0
                for k in range(3):
                    val_eff += B_eff[k, r] * C0_B_eff[k, c]
                    val_b += B_b[k, r] * C2_B_b[k, c]
                K_loc[r, c] += (val_eff + val_b) * detw

        if has_plasticity:
            # Membrane-bending coupling:
            # B_eff.T @ C1 @ B_b + B_b.T @ C1.T @ B_eff.
            # Layered material tangents make C1 symmetric, while a general
            # pre-integrated section may legitimately supply a non-symmetric
            # B coupling block in the symmetric [[A, B], [B.T, D]] law.
            C1_g = C1[g]
            C1_B_b = np.zeros((3, n_dof))
            C1_T_B_eff = np.zeros((3, n_dof))
            for r in range(3):
                for c in range(n_dof):
                    val_b = 0.0
                    val_eff = 0.0
                    for k in range(3):
                        val_b += C1_g[r, k] * B_b[k, c]
                        val_eff += C1_g[k, r] * B_eff[k, c]
                    C1_B_b[r, c] = val_b
                    C1_T_B_eff[r, c] = val_eff

            for r in range(n_dof):
                for c in range(n_dof):
                    forward = 0.0
                    reverse = 0.0
                    for k in range(3):
                        forward += B_eff[k, r] * C1_B_b[k, c]
                        reverse += B_b[k, r] * C1_T_B_eff[k, c]
                    K_loc[r, c] += (forward + reverse) * detw

        # Geometric initial stress stiffness
        N00 = N_g[0]
        N11 = N_g[1]
        N01 = N_g[2]

        N_Gw_0 = N00 * Gw[0] + N01 * Gw[1]
        N_Gw_1 = N01 * Gw[0] + N11 * Gw[1]
        for r in range(n_dof):
            for c in range(n_dof):
                K_loc[r, c] += (Gw[0, r] * N_Gw_0[c] + Gw[1, r] * N_Gw_1[c]) * detw
                K_loc[r, c] += B_d[0, r] * (drilling_stiffness * B_d[0, c]) * detw

    # Shear stiffness and force contribution
    for g in range(len(detw_shear_all)):
        detw_s = detw_shear_all[g]
        B_s = B_s_all[g]

        # F_loc += K_s @ u_loc
        Bs_u_0 = 0.0
        Bs_u_1 = 0.0
        for i in range(n_dof):
            Bs_u_0 += B_s[0, i] * u_loc[i]
            Bs_u_1 += B_s[1, i] * u_loc[i]

        f_s_0 = D_shear[0, 0] * Bs_u_0 + D_shear[0, 1] * Bs_u_1
        f_s_1 = D_shear[1, 0] * Bs_u_0 + D_shear[1, 1] * Bs_u_1

        for i in range(n_dof):
            F_loc[i] += (B_s[0, i] * f_s_0 + B_s[1, i] * f_s_1) * detw_s

        if not tangent:
            continue

        # K_loc += K_s
        C_s = np.zeros((2, n_dof))
        for r in range(2):
            for c in range(n_dof):
                C_s[r, c] = D_shear[r, 0] * B_s[0, c] + D_shear[r, 1] * B_s[1, c]

        for r in range(n_dof):
            for c in range(n_dof):
                K_loc[r, c] += (B_s[0, r] * C_s[0, c] + B_s[1, r] * C_s[1, c]) * detw_s

    return F_loc, K_loc


def _integrate_nonlinear_tangent_components(
    u_loc: np.ndarray,
    N_res: np.ndarray,
    M_res: np.ndarray,
    C0: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    B_m_all: np.ndarray,
    B_b_all: np.ndarray,
    B_d_all: np.ndarray,
    Gw_all: np.ndarray,
    detw_all: np.ndarray,
    B_s_all: np.ndarray,
    detw_shear_all: np.ndarray,
    D_shear: np.ndarray,
    drilling_stiffness: float,
    has_membrane_bending_coupling: bool,
    n_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate constitutive and stress-Hessian tangents independently.

    The two calls deliberately reuse :func:`_jit_integrate_nonlinear_response`
    rather than recovering one component as ``total - other``.  The material
    call carries the exact current algorithmic moduli, transverse shear and
    drilling terms with zero resultants.  The geometric call carries the exact
    tension-positive membrane resultants with every constitutive/numerical
    modulus set to zero.  Consequently both pieces use the same Gauss order,
    von-Karman ``B_eff`` construction and arithmetic kernel as the ordinary
    nonlinear tangent.
    """

    zero_resultants = np.zeros_like(np.asarray(N_res, dtype=float))
    _unused_force, material = _jit_integrate_nonlinear_response(
        np.asarray(u_loc, dtype=float),
        zero_resultants,
        np.zeros_like(np.asarray(M_res, dtype=float)),
        np.asarray(C0, dtype=float),
        np.asarray(C1, dtype=float),
        np.asarray(C2, dtype=float),
        np.asarray(B_m_all, dtype=float),
        np.asarray(B_b_all, dtype=float),
        np.asarray(B_d_all, dtype=float),
        np.asarray(Gw_all, dtype=float),
        np.asarray(detw_all, dtype=float),
        np.asarray(B_s_all, dtype=float),
        np.asarray(detw_shear_all, dtype=float),
        np.asarray(D_shear, dtype=float),
        float(drilling_stiffness),
        True,
        bool(has_membrane_bending_coupling),
        int(n_dof),
    )
    zero_C0 = np.zeros_like(np.asarray(C0, dtype=float))
    zero_C1 = np.zeros_like(np.asarray(C1, dtype=float))
    zero_C2 = np.zeros_like(np.asarray(C2, dtype=float))
    zero_shear = np.zeros_like(np.asarray(D_shear, dtype=float))
    _unused_force, geometric = _jit_integrate_nonlinear_response(
        np.asarray(u_loc, dtype=float),
        np.asarray(N_res, dtype=float),
        np.asarray(M_res, dtype=float),
        zero_C0,
        zero_C1,
        zero_C2,
        np.asarray(B_m_all, dtype=float),
        np.asarray(B_b_all, dtype=float),
        np.asarray(B_d_all, dtype=float),
        np.asarray(Gw_all, dtype=float),
        np.asarray(detw_all, dtype=float),
        np.asarray(B_s_all, dtype=float),
        np.asarray(detw_shear_all, dtype=float),
        zero_shear,
        0.0,
        True,
        False,
        int(n_dof),
    )
    return material, geometric


class ShellElement(Element):
    """3/6-node triangular and 4/8-node quadrilateral Mindlin-Reissner shell element."""

    TRI_GAUSS_POINTS_1 = _immutable_float64_array(
        [[1.0 / 3.0, 1.0 / 3.0]]
    )
    TRI_GAUSS_WEIGHTS_1 = _immutable_float64_array([0.5])

    TRI_GAUSS_POINTS_3 = _immutable_float64_array(
        [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
    )
    TRI_GAUSS_WEIGHTS_3 = _immutable_float64_array(
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    )

    _DUNAVANT_A1 = 0.059715871789770
    _DUNAVANT_B1 = 0.470142064105115
    _DUNAVANT_A2 = 0.797426985353087
    _DUNAVANT_B2 = 0.101286507323456
    TRI_GAUSS_POINTS_7 = _immutable_float64_array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [_DUNAVANT_B1, _DUNAVANT_B1],
            [_DUNAVANT_A1, _DUNAVANT_B1],
            [_DUNAVANT_B1, _DUNAVANT_A1],
            [_DUNAVANT_B2, _DUNAVANT_B2],
            [_DUNAVANT_A2, _DUNAVANT_B2],
            [_DUNAVANT_B2, _DUNAVANT_A2],
        ],
    )
    TRI_GAUSS_WEIGHTS_7 = _immutable_float64_array(
        [
            0.1125,
            0.066197076394253,
            0.066197076394253,
            0.066197076394253,
            0.062969590272414,
            0.062969590272414,
            0.062969590272414,
        ],
    )

    GAUSS_POINTS_1x1 = _immutable_float64_array([[0.0, 0.0]])
    GAUSS_WEIGHTS_1x1 = _immutable_float64_array([4.0])

    GAUSS_POINTS_2x2 = _immutable_float64_array(
        np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        / np.sqrt(3.0)
    )
    GAUSS_WEIGHTS_2x2 = _immutable_float64_array(np.ones(4, dtype=np.float64))

    GAUSS_POINTS_3x3 = _immutable_float64_array(
        [
            [-np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)],
            [0.0, -np.sqrt(3.0 / 5.0)],
            [np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)],
            [-np.sqrt(3.0 / 5.0), 0.0],
            [0.0, 0.0],
            [np.sqrt(3.0 / 5.0), 0.0],
            [-np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)],
            [0.0, np.sqrt(3.0 / 5.0)],
            [np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)],
        ],
    )
    GAUSS_WEIGHTS_3x3 = _immutable_float64_array(
        [
            25.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
            40.0 / 81.0,
            64.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
        ],
    )

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        thickness: float = 0.01,
        drilling_stabilization: float = 1.0e-3,
        reduced_integration: bool = False,
        hourglass_stabilization: float = 1.0e-8,
        material_direction: Optional[np.ndarray] = None,
        material_angle_deg: float = 0.0,
        shell_section: Optional[Any] = None,
    ):
        super().__init__(element_id, node_ids, material_name)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"Shell element {element_id} has repeated node ids")
        if self.__class__ is ShellElement and len(node_ids) == 4:
            warnings.warn(
                "Legacy four-node ShellElement is a temporary rollback path; "
                "use create_shell_element() or QualifiedE4PLShellElement. The "
                "legacy Q4 implementation is available through the 0.4.x "
                "burn-in line and is scheduled for removal no earlier than 0.5.0.",
                LegacyQ4DeprecationWarning,
                stacklevel=2,
            )
        self.thickness = float(thickness)
        self.drilling_stabilization = float(drilling_stabilization)
        self.reduced_integration = reduced_integration
        self.hourglass_stabilization = float(hourglass_stabilization)
        self.material_angle_deg = float(material_angle_deg)
        if not np.isfinite(self.material_angle_deg):
            raise ValueError("material_angle_deg must be finite")
        if material_direction is None:
            self.material_direction = None
        else:
            direction = np.asarray(material_direction, dtype=float).reshape(-1)
            if direction.size != 3 or not np.all(np.isfinite(direction)):
                raise ValueError("material_direction must be a finite 3-vector")
            if float(np.linalg.norm(direction)) <= _SMALL:
                raise ValueError("material_direction must be non-zero")
            self.material_direction = direction.copy()
        self.shell_section = coerce_generalized_shell_section(shell_section)
        self._is_3node = len(node_ids) == 3
        self._is_6node = len(node_ids) == 6
        self._is_8node = len(node_ids) == 8
        self._is_4node = len(node_ids) == 4
        self._is_triangular = self._is_3node or self._is_6node
        self._is_quadrilateral = self._is_4node or self._is_8node
        if not (self._is_triangular or self._is_quadrilateral):
            raise ValueError(f"ShellElement requires 3, 4, 6 or 8 nodes, got {len(node_ids)}")

    def _material_angle(self, local_frame: np.ndarray) -> float:
        """Material 1-axis angle in the supplied element-local frame."""

        angle = np.deg2rad(self.material_angle_deg)
        if self.material_direction is None:
            return float(angle)
        frame = np.asarray(local_frame, dtype=float).reshape(3, 3)
        direction = np.asarray(self.material_direction, dtype=float)
        components = np.array(
            [float(np.dot(direction, frame[:, 0])), float(np.dot(direction, frame[:, 1]))],
            dtype=float,
        )
        norm = float(np.linalg.norm(components))
        if norm <= 1.0e-10 * max(float(np.linalg.norm(direction)), 1.0):
            raise ValueError(
                f"Shell element {self.element_id} material_direction is parallel to "
                "the shell normal and cannot define an in-plane material axis."
            )
        return float(np.arctan2(components[1], components[0]) + angle)

    def _generalized_section_in_frame(
        self,
        local_frame: np.ndarray,
    ) -> Optional[GeneralizedShellSection]:
        """Return the pre-integrated section rotated into shell-local axes."""

        if self.shell_section is None:
            return None
        return self.shell_section.rotated(self._material_angle(local_frame))

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "thickness": float(self.thickness),
                "material_angle_deg": float(self.material_angle_deg),
                "material_direction": (
                    None
                    if self.material_direction is None
                    else np.asarray(self.material_direction, dtype=float).tolist()
                ),
                "shell_section": (
                    None if self.shell_section is None else self.shell_section.to_dict()
                ),
            }
        )
        return payload

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def gauss_points(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_POINTS_3
        if self._is_6node:
            return self.TRI_GAUSS_POINTS_7
        if self._is_8node and self.reduced_integration:
            return self.GAUSS_POINTS_2x2
        return self.GAUSS_POINTS_3x3 if self._is_8node else self.GAUSS_POINTS_2x2

    @property
    def gauss_weights(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_WEIGHTS_3
        if self._is_6node:
            return self.TRI_GAUSS_WEIGHTS_7
        if self._is_8node and self.reduced_integration:
            return self.GAUSS_WEIGHTS_2x2
        return self.GAUSS_WEIGHTS_3x3 if self._is_8node else self.GAUSS_WEIGHTS_2x2

    @property
    def shear_gauss_points(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_POINTS_1
        if self._is_6node:
            return self.TRI_GAUSS_POINTS_3
        return self.GAUSS_POINTS_2x2 if self._is_8node else self.GAUSS_POINTS_1x1

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_WEIGHTS_1
        if self._is_6node:
            return self.TRI_GAUSS_WEIGHTS_3
        return self.GAUSS_WEIGHTS_2x2 if self._is_8node else self.GAUSS_WEIGHTS_1x1

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((self.num_nodes, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Shell element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        return coords

    def compute_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._is_3node:
            return self._compute_3node_shape_functions(xi, eta)
        if self._is_6node:
            return self._compute_6node_shape_functions(xi, eta)
        if self._is_4node:
            return self._compute_4node_shape_functions(xi, eta)
        return self._compute_8node_shape_functions(xi, eta)

    def _compute_3node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_3node_shape_functions(xi, eta)

    def _compute_4node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_4node_shape_functions(xi, eta)

    def _compute_6node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_6node_shape_functions(xi, eta)

    def _compute_8node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_8node_shape_functions(xi, eta)

    def compute_jacobian(self, coords: np.ndarray, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> np.ndarray:
        return np.array([coords.T @ dN_dxi, coords.T @ dN_deta], dtype=float)

    @staticmethod
    def _normalize(vector: np.ndarray) -> Tuple[np.ndarray, float]:
        norm = float(np.sqrt(vector @ vector))
        if norm < _SMALL:
            return np.zeros(3, dtype=float), norm
        return vector / norm, norm

    def _fallback_edge_direction(self, coords: np.ndarray, normal: np.ndarray) -> np.ndarray:
        candidate_edges = []
        if coords.shape[0] >= 2:
            candidate_edges.append(coords[1] - coords[0])
        if coords.shape[0] >= 3:
            candidate_edges.append(coords[2] - coords[0])
            candidate_edges.append(coords[2] - coords[1])
        if coords.shape[0] >= 4:
            candidate_edges.append(coords[2] - coords[3])
            candidate_edges.append(coords[3] - coords[0])
            candidate_edges.append(coords[2] - coords[1])
        for edge in candidate_edges:
            projected = edge - np.dot(edge, normal) * normal
            e1, n = self._normalize(projected)
            if n > _SMALL:
                return e1
        raise ValueError(f"Shell element {self.element_id} has no valid in-plane direction")

    def _local_frame_and_derivatives(
        self,
        coords: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        J = self.compute_jacobian(coords, dN_dxi, dN_deta)
        tangent_xi = J[0]
        tangent_eta = J[1]
        e3, det_j = self._normalize(_cross3(tangent_xi, tangent_eta))
        if det_j < _SMALL:
            raise ValueError(f"Shell element {self.element_id} has a near-zero surface Jacobian")

        e1_raw = tangent_xi - np.dot(tangent_xi, e3) * e3
        e1, e1_norm = self._normalize(e1_raw)
        if e1_norm < _SMALL:
            e1 = self._fallback_edge_direction(coords, e3)
        e2, e2_norm = self._normalize(_cross3(e3, e1))
        if e2_norm < _SMALL:
            raise ValueError(f"Shell element {self.element_id} has an invalid local y direction")
        e1, _ = self._normalize(_cross3(e2, e3))
        R = np.column_stack((e1, e2, e3))

        J_local = np.array(
            [
                [np.dot(tangent_xi, e1), np.dot(tangent_xi, e2)],
                [np.dot(tangent_eta, e1), np.dot(tangent_eta, e2)],
            ],
            dtype=float,
        )
        try:
            inv_j_local, _ = _inv2(J_local)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular local Jacobian") from exc

        dN_dx = inv_j_local[0, 0] * dN_dxi + inv_j_local[0, 1] * dN_deta
        dN_dy = inv_j_local[1, 0] * dN_dxi + inv_j_local[1, 1] * dN_deta
        return R, dN_dx, dN_dy, det_j

    def _local_dof_transform(self, R: np.ndarray) -> np.ndarray:
        n_blocks = 2 * self.num_nodes
        T = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        blocks = T.reshape(n_blocks, 3, n_blocks, 3)
        indices = np.arange(n_blocks)
        blocks[indices, :, indices, :] = R.T
        return T

    def _build_shell_b_matrices(
        self,
        N: np.ndarray,
        dN_dx: np.ndarray,
        dN_dy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B_m = np.zeros((3, self.total_dofs), dtype=float)
        B_b = np.zeros((3, self.total_dofs), dtype=float)
        B_s = np.zeros((2, self.total_dofs), dtype=float)

        B_m[0, 0::6] = dN_dx
        B_m[1, 1::6] = dN_dy
        B_m[2, 0::6] = dN_dy
        B_m[2, 1::6] = dN_dx

        B_b[0, 4::6] = dN_dx
        B_b[1, 3::6] = -dN_dy
        B_b[2, 4::6] = dN_dy
        B_b[2, 3::6] = -dN_dx

        B_s[0, 2::6] = dN_dx
        B_s[0, 4::6] = N
        B_s[1, 2::6] = dN_dy
        B_s[1, 3::6] = -N
        return B_m, B_b, B_s

    def _build_drilling_b_matrix(
        self,
        N: np.ndarray,
        dN_dx: np.ndarray,
        dN_dy: np.ndarray,
    ) -> np.ndarray:
        """
        Build a small drilling stabilization strain.

        The stabilized quantity is theta_z - 0.5 * (dv/dx - du/dy), so a
        physical rigid rotation about the shell normal has exactly zero energy.
        """
        B_d = np.zeros((1, self.total_dofs), dtype=float)
        B_d[0, 0::6] = 0.5 * dN_dy
        B_d[0, 1::6] = -0.5 * dN_dx
        B_d[0, 5::6] = N
        return B_d

    def _reference_center(self) -> Tuple[float, float]:
        if self._is_triangular:
            return 1.0 / 3.0, 1.0 / 3.0
        return 0.0, 0.0

    # MITC4 assumed natural transverse shear (4-node elements only).
    #
    # Covariant shear strains are sampled where they are exact for pure
    # bending, at the edge midpoints A(0,-1), B(1,0), C(0,1), D(-1,0):
    #
    #     gamma_xi(xi, eta)  = (1-eta)/2 * gamma_xi|A  + (1+eta)/2 * gamma_xi|C
    #     gamma_eta(xi, eta) = (1-xi)/2  * gamma_eta|D + (1+xi)/2  * gamma_eta|B
    #
    # with gamma_xi = dw/dxi + x_,xi * theta_y - y_,xi * theta_x in a fixed
    # element-plane frame, then mapped to Cartesian shear through the inverse
    # in-plane Jacobian at the integration point.  The element geometry is
    # treated as a flat facet in the frame evaluated at the element centre.
    _MITC4_SAMPLE_POINTS = {"A": (0.0, -1.0), "B": (1.0, 0.0), "C": (0.0, 1.0), "D": (-1.0, 0.0)}

    def _center_frame(self, coords: np.ndarray) -> np.ndarray:
        xi, eta = self._reference_center()
        _, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
        R, _, _, _ = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
        return R

    def _tri3_assumed_shear_b_matrix(self, coords: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Constant assumed-shear field for the 3-node triangle.

        The row space is evaluated in the centroid frame and integrated once
        over the triangular area.  This is the constant shear-gap part used by
        DSG3/MITC3-style triangles: rigid rotations satisfy
        ``dw/dx + theta_y = 0`` and ``dw/dy - theta_x = 0`` exactly, while a
        constant transverse-shear patch is reproduced without the locking-prone
        fully integrated linear Mindlin shear interpolation.
        """
        if not self._is_3node:
            raise ValueError("_tri3_assumed_shear_b_matrix is only valid for 3-node shells")
        planar = coords @ R[:, :2]
        r, s = self._reference_center()
        N, dN_dr, dN_ds = self.compute_shape_functions(r, s)
        J2 = np.array(
            [
                [float(dN_dr @ planar[:, 0]), float(dN_dr @ planar[:, 1])],
                [float(dN_ds @ planar[:, 0]), float(dN_ds @ planar[:, 1])],
            ],
            dtype=float,
        )
        try:
            inv_j2, det_j = _inv2(J2)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular triangular shear Jacobian") from exc
        dN_dx = inv_j2[0, 0] * dN_dr + inv_j2[0, 1] * dN_ds
        dN_dy = inv_j2[1, 0] * dN_dr + inv_j2[1, 1] * dN_ds
        _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
        return B_s, abs(float(det_j))

    def _mitc4_shear_samples(self, coords: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Return in-plane node coordinates and covariant shear rows at A-D."""
        planar = coords @ R[:, :2]
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, (xi, eta) in self._MITC4_SAMPLE_POINTS.items():
            N, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
            x_xi = float(dN_dxi @ planar[:, 0])
            y_xi = float(dN_dxi @ planar[:, 1])
            x_eta = float(dN_deta @ planar[:, 0])
            y_eta = float(dN_deta @ planar[:, 1])
            row_xi = np.zeros(self.total_dofs, dtype=float)
            row_eta = np.zeros(self.total_dofs, dtype=float)
            row_xi[2::6] = dN_dxi
            row_xi[3::6] = -N * y_xi
            row_xi[4::6] = N * x_xi
            row_eta[2::6] = dN_deta
            row_eta[3::6] = -N * y_eta
            row_eta[4::6] = N * x_eta
            samples[name] = (row_xi, row_eta)
        return planar, samples

    def _mitc4_shear_b_matrix(
        self,
        planar: np.ndarray,
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
        xi: float,
        eta: float,
    ) -> Tuple[np.ndarray, float]:
        """Assumed-shear B matrix (local Cartesian) and in-plane Jacobian det."""
        _, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
        J2 = np.array(
            [
                [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
            ],
            dtype=float,
        )
        try:
            inv_j2, det_j = _inv2(J2)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular in-plane Jacobian") from exc
        B_covariant = np.vstack(
            [
                0.5 * (1.0 - eta) * samples["A"][0] + 0.5 * (1.0 + eta) * samples["C"][0],
                0.5 * (1.0 - xi) * samples["D"][1] + 0.5 * (1.0 + xi) * samples["B"][1],
            ]
        )
        return inv_j2 @ B_covariant, det_j

    def _rigid_body_mode_matrix(self, coords: np.ndarray) -> np.ndarray:
        modes = np.zeros((self.total_dofs, 6), dtype=float)
        centroid = np.mean(coords, axis=0)
        axes = np.eye(3, dtype=float)

        for local, coord in enumerate(coords):
            base = local * 6
            r = coord - centroid
            modes[base + 0, 0] = 1.0
            modes[base + 1, 1] = 1.0
            modes[base + 2, 2] = 1.0
            for axis_index, axis in enumerate(axes):
                displacement = _cross3(axis, r)
                modes[base : base + 3, 3 + axis_index] = displacement
                modes[base + 3 : base + 6, 3 + axis_index] = axis

        q, rmat = np.linalg.qr(modes)
        diag = np.abs(np.diag(rmat))
        if diag.size == 0:
            return np.zeros((self.total_dofs, 0), dtype=float)
        keep = diag > max(float(np.max(diag)) * 1.0e-10, _SMALL)
        return q[:, keep]

    def _hourglass_stabilization_matrix(self, K_base: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Small stiffness on non-rigid zero modes outside rigid motion."""
        coefficient = float(getattr(self, "hourglass_stabilization", 0.0))
        stabilized_topology = (self._is_8node and self.reduced_integration) or self._is_triangular
        if not stabilized_topology or coefficient <= 0.0:
            return np.zeros_like(K_base)

        K_sym = 0.5 * (K_base + K_base.T)
        eigvals, eigvecs = np.linalg.eigh(K_sym)
        max_eig = max(float(np.max(np.abs(eigvals))), 1.0)
        zero_tol = 1.0e-9 * max_eig
        zero_mask = np.abs(eigvals) < zero_tol
        if int(np.sum(zero_mask)) <= 6:
            return np.zeros_like(K_base)

        zero_space = eigvecs[:, zero_mask]
        rigid = self._rigid_body_mode_matrix(coords)
        if rigid.size:
            zero_space = zero_space - rigid @ (rigid.T @ zero_space)

        u, singular_values, _ = np.linalg.svd(zero_space, full_matrices=False)
        if singular_values.size == 0:
            return np.zeros_like(K_base)
        keep = singular_values > 1.0e-8
        if not np.any(keep):
            return np.zeros_like(K_base)

        hourglass_modes = u[:, keep]
        stiffness_scale = coefficient * max_eig
        K_hg = stiffness_scale * (hourglass_modes @ hourglass_modes.T)
        return 0.5 * (K_hg + K_hg.T)

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        generalized_section = self.shell_section is not None
        h = self.thickness
        if generalized_section:
            drilling_modulus = None
        else:
            _elastic_symmetry(material)
            compliance = _elastic_compliance(material)
            drilling_modulus = 1.0 / float(compliance[5, 5])

        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            B_m, B_b, _ = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            B_d = self._build_drilling_b_matrix(N, dN_dx, dN_dy)
            if generalized_section:
                section_local = self._generalized_section_in_frame(R)
                assert section_local is not None
                K_local = (
                    B_m.T @ section_local.A @ B_m
                    + B_m.T @ section_local.B @ B_b
                    + B_b.T @ section_local.B.T @ B_m
                    + B_b.T @ section_local.D @ B_b
                ) * det_j * weight
                drilling_stiffness = (
                    float(section_local.A[2, 2])
                    * getattr(self, "drilling_stabilization", 1.0e-3)
                )
            else:
                Q_local, _G_local, _strain_transform, _stress_transform = (
                    _shell_material_matrices(
                        material,
                        self._material_angle(R),
                    )
                )
                D_membrane = h * Q_local
                D_bending = h**3 / 12.0 * Q_local
                K_local = (
                    B_m.T @ D_membrane @ B_m
                    + B_b.T @ D_bending @ B_b
                ) * det_j * weight
                drilling_stiffness = (
                    float(drilling_modulus)
                    * h
                    * getattr(self, "drilling_stabilization", 1.0e-3)
                )
            K_local += (B_d.T @ (drilling_stiffness * np.eye(1)) @ B_d) * det_j * weight
            K += T.T @ K_local @ T

        if self._is_4node:
            R = self._center_frame(coords)
            if generalized_section:
                section_local = self._generalized_section_in_frame(R)
                assert section_local is not None
                D_shear = section_local.As
            else:
                _Q_local, G_local, _strain_transform, _stress_transform = (
                    _shell_material_matrices(
                        material,
                        self._material_angle(R),
                    )
                )
                D_shear = (5.0 / 6.0) * h * G_local
            T = self._local_dof_transform(R)
            planar, samples = self._mitc4_shear_samples(coords, R)
            for (xi, eta), weight in zip(self.GAUSS_POINTS_2x2, self.GAUSS_WEIGHTS_2x2):
                B_s, det_j = self._mitc4_shear_b_matrix(planar, samples, float(xi), float(eta))
                K_local = (B_s.T @ D_shear @ B_s) * det_j * weight
                K += T.T @ K_local @ T
        elif self._is_3node:
            R = self._center_frame(coords)
            if generalized_section:
                section_local = self._generalized_section_in_frame(R)
                assert section_local is not None
                D_shear = section_local.As
            else:
                _Q_local, G_local, _strain_transform, _stress_transform = (
                    _shell_material_matrices(
                        material,
                        self._material_angle(R),
                    )
                )
                D_shear = (5.0 / 6.0) * h * G_local
            T = self._local_dof_transform(R)
            B_s, det_j = self._tri3_assumed_shear_b_matrix(coords, R)
            K_local = (B_s.T @ D_shear @ B_s) * det_j * float(np.sum(self.shear_gauss_weights))
            K += T.T @ K_local @ T
        else:
            for (xi, eta), weight in zip(self.shear_gauss_points, self.shear_gauss_weights):
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
                if generalized_section:
                    section_local = self._generalized_section_in_frame(R)
                    assert section_local is not None
                    D_shear = section_local.As
                else:
                    _Q_local, G_local, _strain_transform, _stress_transform = (
                        _shell_material_matrices(
                            material,
                            self._material_angle(R),
                        )
                    )
                    D_shear = (5.0 / 6.0) * h * G_local
                T = self._local_dof_transform(R)
                _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
                K_local = (B_s.T @ D_shear @ B_s) * det_j * weight
                K += T.T @ K_local @ T

        self._hourglass_stiffness_matrix = self._hourglass_stabilization_matrix(K, coords)
        K += self._hourglass_stiffness_matrix

        self._stiffness_matrix = K
        return K

    def compute_mass_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        *,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        raw_density = _guarded_observe_attribute(
            material,
            "density",
            post_observation=_qualified_runtime_post_observation,
        )
        rho = _guarded_float(
            raw_density,
            post_observation=_qualified_runtime_post_observation,
        )
        h = self.thickness
        section_mass = (
            None
            if self.shell_section is None
            else _guarded_observe_attribute(
                self.shell_section,
                "mass_per_area",
                post_observation=_qualified_runtime_post_observation,
            )
        )
        mass_per_area = (
            _guarded_float(
                section_mass,
                post_observation=_qualified_runtime_post_observation,
            )
            if section_mass is not None
            else rho * h
        )
        section_rotary_inertia = (
            None
            if self.shell_section is None
            else _guarded_observe_attribute(
                self.shell_section,
                "rotary_inertia_per_area",
                post_observation=_qualified_runtime_post_observation,
            )
        )
        rotary_inertia_per_area = (
            _guarded_float(
                section_rotary_inertia,
                post_observation=_qualified_runtime_post_observation,
            )
            if section_rotary_inertia is not None
            else rho * h**3 / 12.0
        )
        M = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        if self._is_8node and self.reduced_integration:
            area = 0.0
            for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
                _N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                _R, _dN_dx, _dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
                area += float(det_j) * float(weight)
            translational = mass_per_area * max(area, 0.0) / max(self.num_nodes, 1)
            rotational = (
                rotary_inertia_per_area
                * max(area, 0.0)
                / max(self.num_nodes, 1)
            )
            for i in range(self.num_nodes):
                base = 6 * i
                M[base + 0, base + 0] = translational
                M[base + 1, base + 1] = translational
                M[base + 2, base + 2] = translational
                M[base + 3, base + 3] = rotational
                M[base + 4, base + 4] = rotational
                M[base + 5, base + 5] = rotational
            self._mass_matrix = M
            return M
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, _, _, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            M_local = np.zeros_like(M)
            outer_n = np.outer(N, N) * det_j * weight
            translational = mass_per_area * outer_n
            rotational = rotary_inertia_per_area * outer_n
            for d in range(3):
                M_local[d::6, d::6] += translational
                M_local[3 + d::6, 3 + d::6] += rotational
            M += T.T @ M_local @ T
        self._mass_matrix = M
        return M

    @staticmethod
    def _membrane_compression_from_state(state: Optional[Any]) -> Tuple[float, float, float]:
        """Return local membrane resultants with compression-positive convention."""
        if state is None or not isinstance(state, dict):
            return 0.0, 0.0, 0.0

        if "membrane_compression" in state:
            values = np.asarray(state["membrane_compression"], dtype=float).reshape(-1)
            if values.size >= 3:
                return float(values[0]), float(values[1]), float(values[2])
        if "membrane_forces" in state:
            values = np.asarray(state["membrane_forces"], dtype=float).reshape(-1)
            if values.size >= 3:
                return -float(values[0]), -float(values[1]), -float(values[2])

        compression_x = state.get("membrane_compression_x", state.get("Nx_compression"))
        compression_y = state.get("membrane_compression_y", state.get("Ny_compression"))
        compression_xy = state.get("membrane_compression_xy", state.get("Nxy_compression"))
        if compression_x is not None or compression_y is not None or compression_xy is not None:
            return (
                float(compression_x or 0.0),
                float(compression_y or 0.0),
                float(compression_xy or 0.0),
            )

        force_x = state.get("membrane_force_x", state.get("Nx"))
        force_y = state.get("membrane_force_y", state.get("Ny"))
        force_xy = state.get("membrane_force_xy", state.get("Nxy"))
        return (
            -float(force_x or 0.0),
            -float(force_y or 0.0),
            -float(force_xy or 0.0),
        )

    @classmethod
    def _membrane_compression_samples(
        cls,
        state: Optional[Any],
        count: int,
    ) -> np.ndarray:
        """Return compression-positive ``[Nx, Ny, Nxy]`` at integration points.

        Uniform legacy state keys remain supported.  Spatially varying
        prestress can be supplied as ``membrane_compression_at_gauss`` or
        tension-positive ``membrane_forces_at_gauss`` with shape ``(n_gp, 3)``.
        The shorter aliases ``membrane_compression`` and ``membrane_forces``
        also accept that two-dimensional form.
        """
        if isinstance(state, dict):
            compression_values = state.get(
                "membrane_compression_at_gauss",
                state.get("membrane_compression"),
            )
            force_values = state.get(
                "membrane_forces_at_gauss",
                state.get("membrane_forces"),
            )
            values = compression_values if compression_values is not None else force_values
            if values is not None:
                samples = np.asarray(values, dtype=float)
                if samples.ndim == 2:
                    if samples.shape != (int(count), 3):
                        raise ValueError(
                            "Gauss-point membrane resultants must have shape "
                            f"({int(count)}, 3), got {samples.shape}."
                        )
                    return samples.copy() if compression_values is not None else -samples

        uniform = np.asarray(cls._membrane_compression_from_state(state), dtype=float)
        return np.repeat(uniform.reshape(1, 3), int(count), axis=0)

    @staticmethod
    def _resultant_samples(
        state: Optional[Any],
        count: int,
        *,
        compression_keys: Tuple[str, ...],
        tension_keys: Tuple[str, ...] = (),
    ) -> np.ndarray:
        """Read uniform or Gauss-point vector resultants with sign conversion."""
        if not isinstance(state, dict):
            return np.zeros((int(count), 3), dtype=float)
        values = None
        compression_positive = True
        for key in compression_keys:
            if key in state:
                values = state[key]
                break
        if values is None:
            for key in tension_keys:
                if key in state:
                    values = state[key]
                    compression_positive = False
                    break
        if values is None:
            return np.zeros((int(count), 3), dtype=float)
        samples = np.asarray(values, dtype=float)
        if samples.ndim == 1 and samples.size == 3:
            samples = np.repeat(samples.reshape(1, 3), int(count), axis=0)
        if samples.shape != (int(count), 3):
            raise ValueError(
                f"Initial-stress resultants must have shape (3,) or ({int(count)}, 3), "
                f"got {samples.shape}."
            )
        return samples.copy() if compression_positive else -samples

    @classmethod
    def _bending_compression_samples(
        cls,
        state: Optional[Any],
        count: int,
    ) -> np.ndarray:
        """Compression-positive first stress moments ``[Mx, My, Mxy]``."""
        return cls._resultant_samples(
            state,
            count,
            compression_keys=(
                "bending_compression_at_gauss",
                "bending_compression",
                "bending_compression_moments_at_gauss",
                "bending_compression_moments",
            ),
            tension_keys=("bending_moments_at_gauss", "bending_moments"),
        )

    @classmethod
    def _stress_second_moment_samples(
        cls,
        state: Optional[Any],
        count: int,
        membrane_compression: np.ndarray,
        thickness: float,
    ) -> np.ndarray:
        """Compression-positive second stress moments, defaulting to ``N h²/12``."""
        explicit = cls._resultant_samples(
            state,
            count,
            compression_keys=(
                "stress_second_moment_at_gauss",
                "stress_second_moment",
                "membrane_compression_second_moment_at_gauss",
                "membrane_compression_second_moment",
            ),
        )
        has_explicit = isinstance(state, dict) and any(
            key in state
            for key in (
                "stress_second_moment_at_gauss",
                "stress_second_moment",
                "membrane_compression_second_moment_at_gauss",
                "membrane_compression_second_moment",
            )
        )
        if has_explicit:
            return explicit
        return np.asarray(membrane_compression, dtype=float) * float(thickness) ** 2 / 12.0

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Shell stress-stiffness matrix from membrane resultants.

        The state can supply either tension-positive membrane forces
        (``membrane_force_x/y/xy`` or ``membrane_forces``) or
        compression-positive values (``membrane_compression_x/y/xy``).  The
        returned matrix follows the package convention
        ``K phi = lambda KG phi`` with compression destabilizing.

        The total-Lagrangian initial-stress operator uses the Mindlin director
        field ``[u + z*ry, v - z*rx, w]``.  Membrane resultants therefore act
        on all translations, their second stress moment (``N*h**2/12`` for a
        uniform through-thickness membrane stress) acts on ``rx``/``ry``
        gradients, and optional bending resultants provide the signed
        translation-director coupling.  Drilling ``rz`` has no director term.
        """
        compression = self._membrane_compression_samples(state, len(self.gauss_points))
        bending_compression = self._bending_compression_samples(state, len(self.gauss_points))
        second_moment = self._stress_second_moment_samples(
            state,
            len(self.gauss_points),
            compression,
            self.thickness,
        )
        if not np.any(compression) and not np.any(bending_compression) and not np.any(second_moment):
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        KG = np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        for gp_index, ((xi, eta), weight) in enumerate(zip(self.gauss_points, self.gauss_weights)):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            Nx, Ny, Nxy = compression[gp_index]
            N_matrix = np.array([[Nx, Nxy], [Nxy, Ny]], dtype=float)
            Mx, My, Mxy = bending_compression[gp_index]
            M_matrix = np.array([[Mx, Mxy], [Mxy, My]], dtype=float)
            Hx, Hy, Hxy = second_moment[gp_index]
            H_matrix = np.array([[Hx, Hxy], [Hxy, Hy]], dtype=float)
            KG_local = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
            translation_gradients = []
            for component in range(3):
                G = np.zeros((2, self.total_dofs), dtype=float)
                G[0, component::6] = dN_dx
                G[1, component::6] = dN_dy
                translation_gradients.append(G)
                KG_local += G.T @ N_matrix @ G
            G_rx = np.zeros((2, self.total_dofs), dtype=float)
            G_ry = np.zeros((2, self.total_dofs), dtype=float)
            G_rx[0, 3::6] = dN_dx
            G_rx[1, 3::6] = dN_dy
            G_ry[0, 4::6] = dN_dx
            G_ry[1, 4::6] = dN_dy
            KG_local += G_rx.T @ H_matrix @ G_rx
            KG_local += G_ry.T @ H_matrix @ G_ry
            # u(z)=u+z*ry and v(z)=v-z*rx.
            coupling_u_ry = translation_gradients[0].T @ M_matrix @ G_ry
            coupling_v_rx = translation_gradients[1].T @ M_matrix @ G_rx
            KG_local += coupling_u_ry + coupling_u_ry.T
            KG_local -= coupling_v_rx + coupling_v_rx.T
            KG_local *= det_j * float(weight)
            KG += T.T @ KG_local @ T
        return KG

    # ------------------------------------------------------------------
    # Incremental nonlinear response: total-Lagrangian von Karman membrane
    # kinematics in the flat-facet centre frame, layered J2 plane-stress
    # plasticity through the thickness, elastic transverse shear (MITC4 for
    # 4-node, reduced for 8-node) and elastic drilling stabilization.
    # ------------------------------------------------------------------

    def _nonlinear_geometry(self, mesh: "FEMesh") -> Dict[str, Any]:
        """Reference-configuration data, computed once per element."""
        cache = getattr(self, "_nl_cache", None)
        if cache is not None:
            return cache
        coords = self.get_node_coordinates(mesh)
        R0 = self._center_frame(coords)
        T0 = self._local_dof_transform(R0)
        planar = coords @ R0[:, :2]

        gp_data = []
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            J2 = np.array(
                [
                    [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                    [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
                ],
                dtype=float,
            )
            inv_j2, det_j = _inv2(J2)
            dN_dx = inv_j2[0, 0] * dN_dxi + inv_j2[0, 1] * dN_deta
            dN_dy = inv_j2[1, 0] * dN_dxi + inv_j2[1, 1] * dN_deta
            B_m, B_b, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            B_d = self._build_drilling_b_matrix(N, dN_dx, dN_dy)
            Gw = np.zeros((2, self.total_dofs), dtype=float)
            Gw[0, 2::6] = dN_dx
            Gw[1, 2::6] = dN_dy
            gp_data.append(
                {
                    "B_m": B_m,
                    "B_b": B_b,
                    "B_d": B_d,
                    "Gw": Gw,
                    "detw": abs(det_j) * float(weight),
                }
            )

        shear_data = []
        if self._is_4node:
            _, samples = self._mitc4_shear_samples(coords, R0)
            for (xi, eta), weight in zip(self.GAUSS_POINTS_2x2, self.GAUSS_WEIGHTS_2x2):
                B_s, det_j = self._mitc4_shear_b_matrix(planar, samples, float(xi), float(eta))
                shear_data.append({"B_s": B_s, "detw": abs(det_j) * float(weight)})
        elif self._is_3node:
            B_s, det_j = self._tri3_assumed_shear_b_matrix(coords, R0)
            shear_data.append({"B_s": B_s, "detw": det_j * float(np.sum(self.shear_gauss_weights))})
        else:
            for (xi, eta), weight in zip(self.shear_gauss_points, self.shear_gauss_weights):
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                J2 = np.array(
                    [
                        [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                        [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
                    ],
                    dtype=float,
                )
                inv_j2, det_j = _inv2(J2)
                dN_dx = inv_j2[0, 0] * dN_dxi + inv_j2[0, 1] * dN_deta
                dN_dy = inv_j2[1, 0] * dN_dxi + inv_j2[1, 1] * dN_deta
                _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
                shear_data.append({"B_s": B_s, "detw": abs(det_j) * float(weight)})

        n_gp = len(gp_data)
        B_m_all = np.zeros((n_gp, 3, self.total_dofs))
        B_b_all = np.zeros((n_gp, 3, self.total_dofs))
        B_d_all = np.zeros((n_gp, 1, self.total_dofs))
        Gw_all = np.zeros((n_gp, 2, self.total_dofs))
        detw_all = np.zeros(n_gp)
        for g, gp in enumerate(gp_data):
            B_m_all[g] = gp["B_m"]
            B_b_all[g] = gp["B_b"]
            B_d_all[g] = gp["B_d"]
            Gw_all[g] = gp["Gw"]
            detw_all[g] = gp["detw"]

        n_shear = len(shear_data)
        B_s_all = np.zeros((n_shear, 2, self.total_dofs))
        detw_shear_all = np.zeros(n_shear)
        for g, sh in enumerate(shear_data):
            B_s_all[g] = sh["B_s"]
            detw_shear_all[g] = sh["detw"]

        cache = {
            "R0": R0,
            "T0": T0,
            "gp": gp_data,
            "shear": shear_data,
            "B_m_all": B_m_all,
            "B_b_all": B_b_all,
            "B_d_all": B_d_all,
            "Gw_all": Gw_all,
            "detw_all": detw_all,
            "B_s_all": B_s_all,
            "detw_shear_all": detw_shear_all,
        }
        # This is a solver-owned derived cache.  Qualified subclasses reserve
        # the public assignment surface so callers cannot replace a warm
        # nonlinear operator while preserving its cache key.
        object.__setattr__(self, "_nl_cache", cache)
        return cache

    def init_nonlinear_state(self, num_layers: int) -> Dict[str, Any]:
        if self.shell_section is not None:
            return {"generalized_section": True}
        n_points = len(self.gauss_points) * num_layers
        return {
            "plastic_strain": np.zeros((n_points, 3), dtype=float),
            "alpha": np.zeros(n_points, dtype=float),
        }

    def _generalized_section_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any],
        tangent: bool,
        *,
        _return_tangent_components: bool = False,
    ) -> Any:
        """Geometrically nonlinear response for a linear generalized section."""

        if getattr(material, "hardening_curve", None) is not None or getattr(
            material,
            "hill_yield",
            None,
        ) is not None:
            raise ValueError(
                "Generalized shell sections are linear elastic and cannot use "
                "material hardening or Hill plasticity; use a layered section "
                "constitutive model for nonlinear laminate response."
            )
        if isinstance(state, dict) and any(key in state for key in _INITIAL_SHELL_STATE_KEYS):
            raise ValueError(
                "Initial layer stress/prestrain fields are not defined for a "
                "pre-integrated generalized shell section."
            )

        cache = self._nonlinear_geometry(mesh)
        R0 = cache["R0"]
        T0 = cache["T0"]
        u_loc = T0 @ np.asarray(u_elem, dtype=float)
        section = self._generalized_section_in_frame(R0)
        assert section is not None

        n_gp = len(cache["gp"])
        membrane_strain = np.zeros((n_gp, 3), dtype=float)
        curvature = np.zeros((n_gp, 3), dtype=float)
        for index, gp in enumerate(cache["gp"]):
            theta = gp["Gw"] @ u_loc
            membrane_strain[index] = gp["B_m"] @ u_loc + np.array(
                [
                    0.5 * theta[0] ** 2,
                    0.5 * theta[1] ** 2,
                    theta[0] * theta[1],
                ],
                dtype=float,
            )
            curvature[index] = gp["B_b"] @ u_loc

        membrane_resultants = (
            membrane_strain @ section.A.T + curvature @ section.B.T
        )
        bending_resultants = (
            membrane_strain @ section.B + curvature @ section.D.T
        )
        transverse_shear_strain = np.einsum(
            "gij,j->gi",
            cache["B_s_all"],
            u_loc,
        )
        transverse_shear_resultants = (
            transverse_shear_strain @ section.As.T
        )
        C0 = np.broadcast_to(section.A, (n_gp, 3, 3))
        C1 = np.broadcast_to(section.B, (n_gp, 3, 3))
        C2 = np.broadcast_to(section.D, (n_gp, 3, 3))
        drilling_stiffness = (
            float(section.A[2, 2])
            * getattr(self, "drilling_stabilization", 1.0e-3)
        )

        F_loc, K_loc = _jit_integrate_nonlinear_response(
            u_loc,
            membrane_resultants,
            bending_resultants,
            C0,
            C1,
            C2,
            cache["B_m_all"],
            cache["B_b_all"],
            cache["B_d_all"],
            cache["Gw_all"],
            cache["detw_all"],
            cache["B_s_all"],
            cache["detw_shear_all"],
            section.As,
            drilling_stiffness,
            tangent,
            True,
            self.total_dofs,
        )

        K_hg = getattr(self, "_hourglass_stiffness_matrix", None)
        if (
            self._is_8node
            and self.reduced_integration
            and float(getattr(self, "hourglass_stabilization", 0.0)) > 0.0
            and K_hg is None
        ):
            self.compute_stiffness_matrix(mesh, material)
            K_hg = getattr(self, "_hourglass_stiffness_matrix", None)

        F_global = T0.T @ F_loc
        if K_hg is not None:
            F_global = F_global + K_hg @ np.asarray(u_elem, dtype=float)
        K_global: Optional[np.ndarray]
        if tangent:
            K_global = T0.T @ K_loc @ T0
            if K_hg is not None:
                K_global = K_global + K_hg
        else:
            K_global = None

        trial_state: Dict[str, Any] = {
            "generalized_section": True,
            "geometric_nonlinearity": "von_karman",
            "membrane_strain": membrane_strain.copy(),
            "curvature": curvature.copy(),
            "transverse_shear_strain": transverse_shear_strain.copy(),
            "membrane_resultants": membrane_resultants.copy(),
            "bending_resultants": bending_resultants.copy(),
            "transverse_shear_resultants": (
                transverse_shear_resultants.copy()
            ),
            "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
            "transverse_shear_resultant_order": (
                SHELL_TRANSVERSE_SHEAR_ORDER
            ),
            "recovery_scope": "section_resultants_only",
        }
        if _return_tangent_components:
            if K_global is None:
                raise ValueError(
                    "nonlinear tangent components require tangent=True"
                )
            material_local, geometric_local = (
                _integrate_nonlinear_tangent_components(
                    u_loc,
                    membrane_resultants,
                    bending_resultants,
                    C0,
                    C1,
                    C2,
                    cache["B_m_all"],
                    cache["B_b_all"],
                    cache["B_d_all"],
                    cache["Gw_all"],
                    cache["detw_all"],
                    cache["B_s_all"],
                    cache["detw_shear_all"],
                    section.As,
                    drilling_stiffness,
                    True,
                    self.total_dofs,
                )
            )
            material_global = T0.T @ material_local @ T0
            geometric_global = T0.T @ geometric_local @ T0
            if K_hg is not None:
                material_global = material_global + K_hg
            return F_global, K_global, trial_state, {
                "material": material_global,
                "geometric": geometric_global,
            }
        return F_global, K_global, trial_state

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
        *,
        _return_tangent_components: bool = False,
        _qualified_runtime_post_observation: Optional[Callable[[], None]] = None,
    ) -> Any:
        if _return_tangent_components and not tangent:
            raise ValueError("nonlinear tangent components require tangent=True")
        if self.shell_section is not None:
            return self._generalized_section_nonlinear_response(
                mesh,
                material,
                u_elem,
                state,
                tangent,
                _return_tangent_components=_return_tangent_components,
            )
        cache = self._nonlinear_geometry(mesh)
        R0 = cache["R0"]
        T0 = cache["T0"]
        u_loc = T0 @ np.asarray(u_elem, dtype=float)

        symmetry = _elastic_symmetry(
            material,
            _qualified_runtime_post_observation,
        )
        h = self.thickness
        curve = _guarded_observe_attribute(
            material,
            "hardening_curve",
            default=None,
            post_observation=_qualified_runtime_post_observation,
        )
        C_el, shear_elastic, strain_to_material, stress_to_local = _shell_material_matrices(
            material,
            self._material_angle(R0),
            _qualified_runtime_post_observation,
        )
        C_material, _shear_material, _identity_strain, _identity_stress = (
            _shell_material_matrices(
                material,
                0.0,
                _qualified_runtime_post_observation,
            )
        )
        drilling_modulus = 1.0 / float(
            _elastic_compliance(
                material,
                _qualified_runtime_post_observation,
            )[5, 5]
        )
        D_shear = shear_elastic * (5.0 / 6.0) * h
        drilling_stiffness = (
            drilling_modulus
            * h
            * getattr(self, "drilling_stabilization", 1.0e-3)
        )
        hill_yield = _guarded_observe_attribute(
            material,
            "hill_yield",
            default=None,
            post_observation=_qualified_runtime_post_observation,
        )
        if symmetry == "orthotropic" and curve is not None and hill_yield is None:
            raise ValueError(
                f"Orthotropic material {getattr(material, 'name', '<unnamed>')!r} "
                "requires hill_yield when hardening_curve is active."
            )
        hill_plasticity = symmetry == "orthotropic" and hill_yield is not None
        constitutive_plasticity = curve is not None or hill_plasticity

        n_gp = len(cache["gp"])
        z_layers, w_layers = lobatto_layers(num_layers, h)

        if state is None:
            state = self.init_nonlinear_state(num_layers)
        initial_state = _copy_state_fields(state, _INITIAL_SHELL_STATE_KEYS)
        has_initial_field = any(
            key in initial_state
            for key in (
                "initial_membrane_stress",
                "initial_bending_stress",
                "initial_membrane_prestrain",
                "initial_curvature_prestrain",
            )
        )

        # Membrane and bending strains at every integration point.
        memb_strain = np.zeros((n_gp, 3), dtype=float)
        curvature = np.zeros((n_gp, 3), dtype=float)
        B_eff_list = []
        theta_list = []
        for g, gp in enumerate(cache["gp"]):
            theta = gp["Gw"] @ u_loc  # (2,) transverse deflection gradients
            B_nl = np.vstack(
                [
                    theta[0] * gp["Gw"][0],
                    theta[1] * gp["Gw"][1],
                    theta[0] * gp["Gw"][1] + theta[1] * gp["Gw"][0],
                ]
            )
            B_eff = gp["B_m"] + B_nl
            memb_strain[g] = gp["B_m"] @ u_loc + np.array(
                [0.5 * theta[0] ** 2, 0.5 * theta[1] ** 2, theta[0] * theta[1]]
            )
            curvature[g] = gp["B_b"] @ u_loc
            B_eff_list.append(B_eff)
            theta_list.append(theta)

        if not constitutive_plasticity and not has_initial_field:
            # Elastic shortcut: resultants and integrated moduli in closed form.
            trial_state = state
            N_res = memb_strain @ (h * C_el).T
            M_res = curvature @ (h**3 / 12.0 * C_el).T
            C0 = np.broadcast_to(h * C_el, (n_gp, 3, 3))
            C1 = np.zeros((n_gp, 3, 3), dtype=float)
            C2 = np.broadcast_to(h**3 / 12.0 * C_el, (n_gp, 3, 3))
        else:
            # Layer strains for all (gp, layer) points at once.  Prescribed
            # residual stress is represented by its elastic strain offset and
            # prestrain is subtracted as an eigenstrain.  Both therefore enter
            # the same return map as the kinematic strain and participate in
            # yielding, hardening, unloading, and the geometric stiffness.
            kinematic_layer_strain = (
                memb_strain[:, None, :] + z_layers[None, :, None] * curvature[:, None, :]
            )
            zero_rows = np.zeros((n_gp, 3), dtype=float)
            membrane_stress = _shell_field_rows(
                initial_state.get("initial_membrane_stress", zero_rows),
                n_gp,
                "initial_membrane_stress",
            )
            bending_stress = _shell_field_rows(
                initial_state.get("initial_bending_stress", zero_rows),
                n_gp,
                "initial_bending_stress",
            )
            membrane_prestrain = _shell_field_rows(
                initial_state.get("initial_membrane_prestrain", zero_rows),
                n_gp,
                "initial_membrane_prestrain",
            )
            curvature_prestrain = _shell_field_rows(
                initial_state.get("initial_curvature_prestrain", zero_rows),
                n_gp,
                "initial_curvature_prestrain",
            )
            initial_stress = (
                membrane_stress[:, None, :]
                + (2.0 * z_layers[None, :, None] / h) * bending_stress[:, None, :]
            )
            eigenstrain = (
                membrane_prestrain[:, None, :]
                + z_layers[None, :, None] * curvature_prestrain[:, None, :]
            )
            if symmetry == "orthotropic":
                initial_stress_material = np.einsum(
                    "ij,glj->gli",
                    np.linalg.inv(stress_to_local),
                    initial_stress,
                )
                kinematic_material = np.einsum(
                    "ij,glj->gli",
                    strain_to_material,
                    kinematic_layer_strain,
                )
                eigenstrain_material = np.einsum(
                    "ij,glj->gli",
                    strain_to_material,
                    eigenstrain,
                )
                stress_strain_offset = np.einsum(
                    "ij,glj->gli",
                    np.linalg.inv(C_material),
                    initial_stress_material,
                )
                layer_strain_material = (
                    kinematic_material - eigenstrain_material + stress_strain_offset
                ).reshape(n_gp * num_layers, 3)
                layer_strain = (
                    kinematic_layer_strain - eigenstrain
                    + np.einsum(
                        "ij,glj->gli",
                        np.linalg.inv(C_el),
                        initial_stress,
                    )
                ).reshape(n_gp * num_layers, 3)
            else:
                stress_strain_offset = np.einsum(
                    "ij,glj->gli",
                    np.linalg.inv(C_el),
                    initial_stress,
                )
                layer_strain = (
                    kinematic_layer_strain - eigenstrain + stress_strain_offset
                ).reshape(n_gp * num_layers, 3)
                layer_strain_material = layer_strain
            if not constitutive_plasticity:
                sigma = layer_strain @ C_el.T
                C_tan = np.broadcast_to(C_el, (n_gp * num_layers, 3, 3)).copy()
                ep_new = np.zeros_like(layer_strain)
                alpha_new = np.zeros(n_gp * num_layers, dtype=float)
                if symmetry == "orthotropic":
                    sigma_material = layer_strain_material @ C_material.T
            elif hill_plasticity:
                from .plasticity import hill48_plane_stress_return_map

                sigma_material, C_tan_material, ep_new, alpha_new = (
                    hill48_plane_stress_return_map(
                        layer_strain_material,
                        state["plastic_strain"],
                        state["alpha"],
                        C_material,
                        hill_yield,
                        curve,
                        compute_tangent=tangent,
                    )
                )
                sigma = sigma_material @ stress_to_local.T
                C_tan = np.einsum(
                    "ij,njk,kl->nil",
                    stress_to_local,
                    C_tan_material,
                    strain_to_material,
                )
            else:
                sigma, C_tan, ep_new, alpha_new = plane_stress_return_map(
                    layer_strain,
                    state["plastic_strain"],
                    state["alpha"],
                    _guarded_float(
                        _guarded_observe_attribute(
                            material,
                            "elastic_modulus",
                            post_observation=_qualified_runtime_post_observation,
                        ),
                        post_observation=_qualified_runtime_post_observation,
                    ),
                    _guarded_float(
                        _guarded_observe_attribute(
                            material,
                            "poisson_ratio",
                            post_observation=_qualified_runtime_post_observation,
                        ),
                        post_observation=_qualified_runtime_post_observation,
                    ),
                    curve,
                    compute_tangent=tangent,
                )
            trial_state = {
                "plastic_strain": ep_new,
                "alpha": alpha_new,
                "layer_strain": layer_strain.copy(),
                "layer_strain_material": layer_strain_material.copy(),
                "kinematic_layer_strain": kinematic_layer_strain.reshape(n_gp * num_layers, 3).copy(),
                **initial_state,
            }

            sigma = sigma.reshape(n_gp, num_layers, 3)
            C_tan = C_tan.reshape(n_gp, num_layers, 3, 3)
            trial_state["layer_stress"] = sigma.reshape(n_gp * num_layers, 3).copy()
            if symmetry == "orthotropic":
                trial_state["layer_stress_material"] = sigma_material.reshape(
                    n_gp * num_layers,
                    3,
                ).copy()
                trial_state["equivalent_stress_measure"] = (
                    "hill48" if hill_yield is not None else "von_mises"
                )

            # Through-thickness resultants and integrated tangent moduli.
            N_res = np.einsum("l,gli->gi", w_layers, sigma)
            M_res = np.einsum("l,l,gli->gi", w_layers, z_layers, sigma)
            C0 = np.einsum("l,glij->gij", w_layers, C_tan)
            C1 = np.einsum("l,l,glij->gij", w_layers, z_layers, C_tan)
            C2 = np.einsum("l,l,l,glij->gij", w_layers, z_layers, z_layers, C_tan)

        n_dof = self.total_dofs
        F_loc, K_loc_temp = _jit_integrate_nonlinear_response(
            u_loc,
            N_res,
            M_res,
            C0,
            C1,
            C2,
            cache["B_m_all"],
            cache["B_b_all"],
            cache["B_d_all"],
            cache["Gw_all"],
            cache["detw_all"],
            cache["B_s_all"],
            cache["detw_shear_all"],
            D_shear,
            drilling_stiffness,
            tangent,
            constitutive_plasticity,
            n_dof,
        )
        K_loc = K_loc_temp if tangent else None

        K_hg = getattr(self, "_hourglass_stiffness_matrix", None)
        if self._is_8node and self.reduced_integration and float(getattr(self, "hourglass_stabilization", 0.0)) > 0.0:
            if K_hg is None:
                self.compute_stiffness_matrix(mesh, material)
                K_hg = getattr(self, "_hourglass_stiffness_matrix", None)

        F_global = T0.T @ F_loc
        if K_hg is not None:
            u_global = np.asarray(u_elem, dtype=float)
            F_global = F_global + K_hg @ u_global

        if not tangent:
            return F_global, None, trial_state
        K_global = T0.T @ K_loc @ T0
        if K_hg is not None:
            K_global = K_global + K_hg
        if _return_tangent_components:
            material_local, geometric_local = (
                _integrate_nonlinear_tangent_components(
                    u_loc,
                    N_res,
                    M_res,
                    C0,
                    C1,
                    C2,
                    cache["B_m_all"],
                    cache["B_b_all"],
                    cache["B_d_all"],
                    cache["Gw_all"],
                    cache["detw_all"],
                    cache["B_s_all"],
                    cache["detw_shear_all"],
                    D_shear,
                    drilling_stiffness,
                    constitutive_plasticity,
                    n_dof,
                )
            )
            material_global = T0.T @ material_local @ T0
            geometric_global = T0.T @ geometric_local @ T0
            if K_hg is not None:
                material_global = material_global + K_hg
            return F_global, K_global, trial_state, {
                "material": material_global,
                "geometric": geometric_global,
            }
        return F_global, K_global, trial_state

    def _compute_generalized_section_results(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        *,
        return_global: bool,
    ) -> Dict[str, Any]:
        """Recover exact generalized strains and resultants.

        A/B/D/As do not contain enough information to reconstruct ply or
        through-thickness surface stresses, so this path intentionally does
        not report von Mises, Hill, or fabricated top/bottom stress fields.
        """

        coords = self.get_node_coordinates(mesh)
        u_elem_global = self._get_element_displacements(mesh, displacements)
        num_ip = len(self.gauss_points)
        membrane_strain = np.zeros((num_ip, 3), dtype=float)
        curvature = np.zeros((num_ip, 3), dtype=float)
        transverse_shear_strain = np.zeros((num_ip, 2), dtype=float)
        membrane_resultants = np.zeros((num_ip, 3), dtype=float)
        bending_resultants = np.zeros((num_ip, 3), dtype=float)
        transverse_shear_resultants = np.zeros((num_ip, 2), dtype=float)

        global_membrane = (
            np.zeros((num_ip, 3, 3), dtype=float) if return_global else None
        )
        global_bending = (
            np.zeros((num_ip, 3, 3), dtype=float) if return_global else None
        )
        global_shear = (
            np.zeros((num_ip, 3), dtype=float) if return_global else None
        )

        mitc_planar = None
        mitc_samples = None
        mitc_u_local = None
        tri3_B_s = None
        tri3_u_local = None
        center_section = None
        if self._is_4node:
            R_center = self._center_frame(coords)
            center_section = self._generalized_section_in_frame(R_center)
            mitc_planar, mitc_samples = self._mitc4_shear_samples(coords, R_center)
            mitc_u_local = self._local_dof_transform(R_center) @ u_elem_global
        elif self._is_3node:
            R_center = self._center_frame(coords)
            center_section = self._generalized_section_in_frame(R_center)
            tri3_B_s, _ = self._tri3_assumed_shear_b_matrix(coords, R_center)
            tri3_u_local = self._local_dof_transform(R_center) @ u_elem_global

        for index, (xi, eta) in enumerate(self.gauss_points):
            N, dN_dxi, dN_deta = self.compute_shape_functions(
                float(xi),
                float(eta),
            )
            R, dN_dx, dN_dy, _ = self._local_frame_and_derivatives(
                coords,
                dN_dxi,
                dN_deta,
            )
            section = self._generalized_section_in_frame(R)
            assert section is not None
            T = self._local_dof_transform(R)
            u_local = T @ u_elem_global
            B_m, B_b, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            strain = B_m @ u_local
            bend = B_b @ u_local
            if self._is_4node:
                B_s_mitc, _ = self._mitc4_shear_b_matrix(
                    mitc_planar,
                    mitc_samples,
                    float(xi),
                    float(eta),
                )
                shear_strain = B_s_mitc @ mitc_u_local
                shear_section = center_section
            elif self._is_3node:
                shear_strain = tri3_B_s @ tri3_u_local
                shear_section = center_section
            else:
                shear_strain = B_s @ u_local
                shear_section = section
            assert shear_section is not None

            force = section.A @ strain + section.B @ bend
            moment = section.B.T @ strain + section.D @ bend
            shear_force = shear_section.As @ shear_strain

            membrane_strain[index] = strain
            curvature[index] = bend
            transverse_shear_strain[index] = shear_strain
            membrane_resultants[index] = force
            bending_resultants[index] = moment
            transverse_shear_resultants[index] = shear_force

            if return_global:
                force_tensor = np.array(
                    [
                        [force[0], force[2], 0.0],
                        [force[2], force[1], 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
                moment_tensor = np.array(
                    [
                        [moment[0], moment[2], 0.0],
                        [moment[2], moment[1], 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
                assert global_membrane is not None
                assert global_bending is not None
                assert global_shear is not None
                global_membrane[index] = R @ force_tensor @ R.T
                global_bending[index] = R @ moment_tensor @ R.T
                global_shear[index] = (
                    shear_force[0] * R[:, 0] + shear_force[1] * R[:, 1]
                )

        results: Dict[str, Any] = {
            "generalized_stress_scope": "section_resultants_only",
            "recovery_scope": "section_resultants_only",
            "physical_stress_available": False,
            "membrane_resultant_order": SHELL_MEMBRANE_VOIGT_ORDER,
            "transverse_shear_resultant_order": SHELL_TRANSVERSE_SHEAR_ORDER,
            "membrane_strain": membrane_strain,
            "curvature": curvature,
            "transverse_shear_strain": transverse_shear_strain,
            "membrane_resultants": membrane_resultants,
            "bending_resultants": bending_resultants,
            "transverse_shear_resultants": transverse_shear_resultants,
        }
        if return_global:
            results.update(
                {
                    "global_membrane_resultant_tensors": global_membrane,
                    "global_bending_resultant_tensors": global_bending,
                    "global_transverse_shear_resultants": global_shear,
                }
            )
        return results

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
        *,
        natural_points=None,
    ) -> Dict[str, Any]:
        if self.shell_section is not None:
            if natural_points is not None:
                raise ValueError('custom stress points are unsupported for generalized shell sections')
            return self._compute_generalized_section_results(
                mesh,
                displacements,
                return_global=return_global,
            )
        coords = self.get_node_coordinates(mesh)
        u_elem_global = self._get_element_displacements(mesh, displacements)
        symmetry = _elastic_symmetry(material)
        h = self.thickness
        kappa = 5.0 / 6.0

        recovery_points=self.gauss_points if natural_points is None else np.asarray(natural_points,dtype=float)
        if natural_points is not None and (recovery_points.ndim!=2 or recovery_points.shape[1]!=2 or len(recovery_points)==0 or not np.all(np.isfinite(recovery_points)) or np.any(np.abs(recovery_points)>1)):
            raise ValueError('natural stress points must be nonempty finite Nx2 in [-1,1]')
        num_ip = len(recovery_points)
        stresses: Dict[str, np.ndarray] = {
            "membrane_xx": np.zeros(num_ip),
            "membrane_yy": np.zeros(num_ip),
            "membrane_xy": np.zeros(num_ip),
            "bending_xx": np.zeros(num_ip),
            "bending_yy": np.zeros(num_ip),
            "bending_xy": np.zeros(num_ip),
            "shear_xz": np.zeros(num_ip),
            "shear_yz": np.zeros(num_ip),
            "von_mises": np.zeros(num_ip),
            "equivalent_stress": np.zeros(num_ip),
            "hill_utilization": np.zeros(num_ip),
        }
        stresses["equivalent_stress_measure"] = (
            "hill48" if symmetry == "orthotropic" and getattr(material, "hill_yield", None) is not None
            else "von_mises"
        )
        if return_global:
            stresses.update({
                "local_xx_top": np.zeros(num_ip),
                "local_yy_top": np.zeros(num_ip),
                "local_zz_top": np.zeros(num_ip),
                "local_xy_top": np.zeros(num_ip),
                "local_yz_top": np.zeros(num_ip),
                "local_xz_top": np.zeros(num_ip),
                "local_xx_bot": np.zeros(num_ip),
                "local_yy_bot": np.zeros(num_ip),
                "local_zz_bot": np.zeros(num_ip),
                "local_xy_bot": np.zeros(num_ip),
                "local_yz_bot": np.zeros(num_ip),
                "local_xz_bot": np.zeros(num_ip),
                "global_xx_top": np.zeros(num_ip),
                "global_yy_top": np.zeros(num_ip),
                "global_zz_top": np.zeros(num_ip),
                "global_xy_top": np.zeros(num_ip),
                "global_yz_top": np.zeros(num_ip),
                "global_xz_top": np.zeros(num_ip),
                "global_xx_bot": np.zeros(num_ip),
                "global_yy_bot": np.zeros(num_ip),
                "global_zz_bot": np.zeros(num_ip),
                "global_xy_bot": np.zeros(num_ip),
                "global_yz_bot": np.zeros(num_ip),
                "global_xz_bot": np.zeros(num_ip),
            })

        mitc_planar = None
        mitc_samples = None
        mitc_u_local = None
        tri3_B_s = None
        tri3_u_local = None
        if self._is_4node:
            R_center = self._center_frame(coords)
            _Q_center, G_center, _strain_center, _stress_center = _shell_material_matrices(
                material,
                self._material_angle(R_center),
            )
            mitc_planar, mitc_samples = self._mitc4_shear_samples(coords, R_center)
            mitc_u_local = self._local_dof_transform(R_center) @ u_elem_global
        elif self._is_3node:
            R_center = self._center_frame(coords)
            _Q_center, G_center, _strain_center, _stress_center = _shell_material_matrices(
                material,
                self._material_angle(R_center),
            )
            tri3_B_s, _ = self._tri3_assumed_shear_b_matrix(coords, R_center)
            tri3_u_local = self._local_dof_transform(R_center) @ u_elem_global

        from .recovery_shell_frames import native_recovery_frames

        frames = native_recovery_frames(self, coords, recovery_points, material)
        for idx, (xi, eta) in enumerate(recovery_points):
            if frames is None:
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                R, dN_dx, dN_dy, _ = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            else:
                N, R, dN_dx, dN_dy = frames[idx]
            Q_local, G_local, _strain_to_material, stress_to_local = _shell_material_matrices(
                material,
                self._material_angle(R),
            )
            T = self._local_dof_transform(R)
            u_local = T @ u_elem_global
            B_m, B_b, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            membrane_strain = B_m @ u_local
            curvature = B_b @ u_local
            if self._is_4node:
                B_s_mitc, _ = self._mitc4_shear_b_matrix(mitc_planar, mitc_samples, float(xi), float(eta))
                shear_strain = B_s_mitc @ mitc_u_local
            elif self._is_3node:
                shear_strain = tri3_B_s @ tri3_u_local
            else:
                shear_strain = B_s @ u_local

            sigma_m = Q_local @ membrane_strain
            moments = (h**3 / 12.0 * Q_local) @ curvature
            sigma_b = 6.0 * moments / max(h**2, _SMALL)
            shear_matrix = G_center if self._is_4node or self._is_3node else G_local
            tau_s = kappa * shear_matrix @ shear_strain

            stresses["membrane_xx"][idx] = sigma_m[0]
            stresses["membrane_yy"][idx] = sigma_m[1]
            stresses["membrane_xy"][idx] = sigma_m[2]
            stresses["bending_xx"][idx] = sigma_b[0]
            stresses["bending_yy"][idx] = sigma_b[1]
            stresses["bending_xy"][idx] = sigma_b[2]
            stresses["shear_xz"][idx] = tau_s[0]
            stresses["shear_yz"][idx] = tau_s[1]

            # Top surface (z = +h/2)
            sigma_x_top = sigma_m[0] + sigma_b[0]
            sigma_y_top = sigma_m[1] + sigma_b[1]
            tau_xy_top = sigma_m[2] + sigma_b[2]
            vm_top = np.sqrt(
                sigma_x_top**2 + sigma_y_top**2 - sigma_x_top * sigma_y_top + 3.0 * (tau_xy_top**2 + tau_s[0]**2 + tau_s[1]**2)
            )

            # Bottom surface (z = -h/2)
            sigma_x_bot = sigma_m[0] - sigma_b[0]
            sigma_y_bot = sigma_m[1] - sigma_b[1]
            tau_xy_bot = sigma_m[2] - sigma_b[2]
            vm_bot = np.sqrt(
                sigma_x_bot**2 + sigma_y_bot**2 - sigma_x_bot * sigma_y_bot + 3.0 * (tau_xy_bot**2 + tau_s[0]**2 + tau_s[1]**2)
            )

            stresses["von_mises"][idx] = max(vm_top, vm_bot)
            hill_yield = getattr(material, "hill_yield", None)
            if hill_yield is not None:
                from .plasticity import hill48_plane_stress_equivalent_stress

                top_material = np.linalg.solve(stress_to_local, np.array(
                    [sigma_x_top, sigma_y_top, tau_xy_top],
                    dtype=float,
                ))
                bottom_material = np.linalg.solve(stress_to_local, np.array(
                    [sigma_x_bot, sigma_y_bot, tau_xy_bot],
                    dtype=float,
                ))
                hill_values = hill48_plane_stress_equivalent_stress(
                    np.vstack((top_material, bottom_material)),
                    hill_yield,
                )
                equivalent = float(np.max(hill_values))
                stresses["equivalent_stress"][idx] = equivalent
                stresses["hill_utilization"][idx] = equivalent / max(float(hill_yield.X), _SMALL)
            else:
                stresses["equivalent_stress"][idx] = stresses["von_mises"][idx]

            if return_global:
                # Top local stress tensor
                sigma_loc_top = np.array([
                    [sigma_x_top, tau_xy_top, tau_s[0]],
                    [tau_xy_top, sigma_y_top, tau_s[1]],
                    [tau_s[0], tau_s[1], 0.0]
                ], dtype=float)
                sigma_glob_top = R @ sigma_loc_top @ R.T

                # Bottom local stress tensor
                sigma_loc_bot = np.array([
                    [sigma_x_bot, tau_xy_bot, tau_s[0]],
                    [tau_xy_bot, sigma_y_bot, tau_s[1]],
                    [tau_s[0], tau_s[1], 0.0]
                ], dtype=float)
                sigma_glob_bot = R @ sigma_loc_bot @ R.T

                # Store local components
                stresses["local_xx_top"][idx] = sigma_x_top
                stresses["local_yy_top"][idx] = sigma_y_top
                stresses["local_zz_top"][idx] = 0.0
                stresses["local_xy_top"][idx] = tau_xy_top
                stresses["local_xz_top"][idx] = tau_s[0]
                stresses["local_yz_top"][idx] = tau_s[1]

                stresses["local_xx_bot"][idx] = sigma_x_bot
                stresses["local_yy_bot"][idx] = sigma_y_bot
                stresses["local_zz_bot"][idx] = 0.0
                stresses["local_xy_bot"][idx] = tau_xy_bot
                stresses["local_xz_bot"][idx] = tau_s[0]
                stresses["local_yz_bot"][idx] = tau_s[1]

                # Store global components
                stresses["global_xx_top"][idx] = sigma_glob_top[0, 0]
                stresses["global_yy_top"][idx] = sigma_glob_top[1, 1]
                stresses["global_zz_top"][idx] = sigma_glob_top[2, 2]
                stresses["global_xy_top"][idx] = sigma_glob_top[0, 1]
                stresses["global_xz_top"][idx] = sigma_glob_top[0, 2]
                stresses["global_yz_top"][idx] = sigma_glob_top[1, 2]

                stresses["global_xx_bot"][idx] = sigma_glob_bot[0, 0]
                stresses["global_yy_bot"][idx] = sigma_glob_bot[1, 1]
                stresses["global_zz_bot"][idx] = sigma_glob_bot[2, 2]
                stresses["global_xy_bot"][idx] = sigma_glob_bot[0, 1]
                stresses["global_xz_bot"][idx] = sigma_glob_bot[0, 2]
                stresses["global_yz_bot"][idx] = sigma_glob_bot[1, 2]
        return stresses


def _publish_shell_element_public_signatures() -> None:
    """Hide private runtime-guard hooks from the stable public API surface."""

    from inspect import signature

    mass_signature = signature(ShellElement.compute_mass_matrix)
    ShellElement.compute_mass_matrix.__signature__ = mass_signature.replace(
        parameters=tuple(
            parameter
            for parameter in mass_signature.parameters.values()
            if parameter.name != "_qualified_runtime_post_observation"
        )
    )
    nonlinear_signature = signature(ShellElement.compute_nonlinear_response)
    public_nonlinear_return = signature(
        Element.compute_nonlinear_response
    ).return_annotation
    ShellElement.compute_nonlinear_response.__signature__ = (
        nonlinear_signature.replace(
            parameters=tuple(
                parameter
                for parameter in nonlinear_signature.parameters.values()
                if parameter.name
                not in {
                    "_return_tangent_components",
                    "_qualified_runtime_post_observation",
                }
            ),
            return_annotation=public_nonlinear_return,
        )
    )


_publish_shell_element_public_signatures()
del _publish_shell_element_public_signatures


def _install_strict_flat_s3_v2_legacy_dispatch_guards() -> None:
    """Reject explicit base-method bypasses on the Stage-1 V2 candidate."""

    from functools import wraps
    from inspect import signature

    guard = _reject_strict_flat_s3_v2_legacy_dispatch
    allowed = {"__init__", "dofs_per_node", "get_node_coordinates", "num_nodes"}
    for name, original in tuple(vars(ShellElement).items()):
        if name in allowed or name.startswith("__"):
            continue

        if type(original) is property:
            def wrap_accessor(
                accessor: Optional[Callable[..., Any]],
                label: str,
                _guard: Callable[[Any, str], None] = guard,
            ) -> Optional[Callable[..., Any]]:
                if accessor is None:
                    return None

                @wraps(accessor)
                def guarded_accessor(self: Any, *args: Any, **kwargs: Any) -> Any:
                    _guard(self, f"ShellElement.{label}")
                    return accessor(self, *args, **kwargs)

                guarded_accessor.__signature__ = signature(accessor)  # type: ignore[attr-defined]
                delattr(guarded_accessor, "__wrapped__")
                return guarded_accessor

            setattr(
                ShellElement,
                name,
                property(
                    wrap_accessor(original.fget, name),
                    wrap_accessor(original.fset, name),
                    wrap_accessor(original.fdel, name),
                    original.__doc__,
                ),
            )
            continue
        if type(original) is not FunctionType:
            continue

        @wraps(original)
        def guarded(
            self: Any,
            *args: Any,
            __name: str = name,
            __original: Callable[..., Any] = original,
            __guard: Callable[[Any, str], None] = guard,
            **kwargs: Any,
        ) -> Any:
            __guard(self, f"ShellElement.{__name}")
            return __original(self, *args, **kwargs)

        guarded.__signature__ = signature(original)  # type: ignore[attr-defined]
        delattr(guarded, "__wrapped__")
        setattr(ShellElement, name, guarded)


_install_strict_flat_s3_v2_legacy_dispatch_guards()
del _install_strict_flat_s3_v2_legacy_dispatch_guards

# Exact dispatch surface for the recovery-only quadrature frame optimization.
# Subclasses and instance/class overrides retain their scalar dispatch.
_SHELL_RECOVERY_FRAME_METHODS = tuple(vars(ShellElement).items())
_SHELL_RECOVERY_BASE_METHODS = tuple(vars(Element).items())
_SHELL_RECOVERY_MATERIAL_METHODS = tuple(vars(_RecoveryIsotropicMaterial).items())
_SHELL_RECOVERY_FRAME_HELPERS = tuple((name, globals()[name]) for name in (
    "_cross3", "_inv2", "_shell_material_matrices", "_elastic_symmetry",
    "_canonical_material_symmetry", "_guarded_observe_attribute", "_guarded_float",
    "_jit_compute_8node_shape_functions",
))


class BeamElement(Element):
    """2-node Timoshenko beam element with 6 DOF per node.

    ``section=`` opts into a coupled generalized section.  The same object may
    be supplied under ``cross_section["generalized_section"]``; a raw matrix
    is accepted under ``cross_section["generalized_stiffness"]``.  Legacy
    scalar cross-section dictionaries retain their historical formulation.
    A generalized stiffness matrix does not define centroidal geometry, so
    its Wagner torsional geometric-stiffness term is omitted by default.
    Supply ``cross_section["geometric_polar_radius_squared"]`` in m^2 to opt in
    explicitly.
    """

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        cross_section: Optional[Dict[str, Any]] = None,
        section: Any = None,
    ):
        super().__init__(element_id, node_ids, material_name)
        if len(node_ids) != 2:
            raise ValueError(f"BeamElement requires 2 nodes, got {len(node_ids)}")
        self.cross_section = cross_section or {}
        self._A = self.cross_section.get("area", 0.01)
        self._Iy = self.cross_section.get("Iy", 1.0e-8)
        self._Iz = self.cross_section.get("Iz", 1.0e-8)
        self._J = self.cross_section.get("J", 1.0e-8)
        self._ky = self.cross_section.get("shear_factor_y", 5.0 / 6.0)
        self._kz = self.cross_section.get("shear_factor_z", 5.0 / 6.0)
        self._orientation = _section_orientation(self.cross_section)
        self._fiber_plasticity = self.cross_section.get("fiber_plasticity")
        self.generalized_section = resolve_generalized_beam_section(
            self.cross_section,
            section,
        )
        if self.generalized_section is not None and self._fiber_plasticity is not None:
            raise ValueError(
                "Generalized beam sections do not support "
                "cross_section['fiber_plasticity']; supply an elastic section "
                "without fiber plasticity."
            )
        if self.generalized_section is not None:
            self._geometric_polar_radius_squared()
        self._geometric_nonlinearity = str(
            self.cross_section.get("geometric_nonlinearity", self.cross_section.get("geometry", "von_karman"))
        ).lower()
        # Optional exact stress-recovery data; estimated from A and I if absent.
        self._c_y = self.cross_section.get("c_y")
        self._c_z = self.cross_section.get("c_z")
        self._torsion_modulus = self.cross_section.get("torsion_modulus")

    def set_linear_nodal_offsets(self, offsets) -> None:
        """Global node-to-centroid rigid arms for the linear static route only."""
        values=np.asarray(offsets,dtype=float)
        if self.num_nodes!=2 or values.shape!=(2,3) or not np.all(np.isfinite(values)):
            raise ValueError('linear beam offsets require two finite global vectors')
        self._linear_nodal_offsets=values.copy()

    def _linear_rigid_arm_map(self) -> np.ndarray:
        arm=np.eye(12)
        for i,d in enumerate(getattr(self,'_linear_nodal_offsets',np.zeros((2,3)))):
            x,y,z=d
            arm[i*6:i*6+3,i*6+3:i*6+6]=np.array([[0,z,-y],[-z,0,x],[y,-x,0]])
        return arm

    def recover_linear_section_forces(self, mesh, displacements, material, stations):
        """Global force/moment on a positive local-x cut; no beam distributed loads."""
        stations=np.asarray(stations,dtype=float)
        if stations.ndim!=2 or stations.shape[1]!=3 or len(stations)==0 or not np.all(np.isfinite(stations)):
            raise ValueError('beam stations must be nonempty finite Nx3 coordinates')
        coords=self.get_node_coordinates(mesh)
        length,transform=self._beam_frame_and_transform(coords)
        local=transform@self._get_element_displacements(mesh,displacements)
        end=self._local_linear_stiffness(length,material)@local
        rotation=_beam_rotation_matrix((coords[1]-coords[0])/length,self._orientation)
        output=[]
        for station in stations:
            distance=float(np.dot(station-coords[0],rotation[:,0]))
            if distance < -1e-6 or distance>length+1e-6 or np.linalg.norm(station-(coords[0]+distance*rotation[:,0]))>1e-6:
                raise ValueError('beam result station is not on physical centroid line')
            force=-end[:3]
            moment=-end[3:6]+np.cross([distance,0,0],end[:3])
            output.append(np.r_[rotation@force,rotation@moment])
        return np.asarray(output)

    def _refuse_offset_route(self, route):
        if np.any(getattr(self,'_linear_nodal_offsets',0)):
            raise ValueError(f'offset beam {route} is not admitted; linear static only')

    def _fiber_distances(self) -> Tuple[float, float]:
        """Extreme fiber distances (local y, local z), estimated when not given."""
        c_y = self._c_y
        c_z = self._c_z
        if c_y is None or c_y <= 0.0:
            c_y = np.sqrt(abs(self._Iz) / max(self._A, _SMALL)) * 2.0
        if c_z is None or c_z <= 0.0:
            c_z = np.sqrt(abs(self._Iy) / max(self._A, _SMALL)) * 2.0
        return max(float(c_y), _SMALL), max(float(c_z), _SMALL)

    def _torsion_section_modulus(self) -> float:
        """Torsional section modulus Wt with tau = T / Wt, estimated when not given."""
        wt = self._torsion_modulus
        if wt is None or wt <= 0.0:
            wt = 2.0 * self._A
        return max(float(wt), _SMALL)

    def _geometric_polar_radius_squared(self) -> float:
        """Return the section polar radius squared used by the Wagner term.

        The legacy scalar section defines this quantity as ``(Iy + Iz) / A``.
        A generalized 6x6 stiffness does not uniquely define any of those
        geometric quantities, even when legacy keys happen to be present for
        mass fallback or other workflows.  Generalized sections therefore
        require an explicitly named geometric value and otherwise omit the
        Wagner term.
        """
        if self.generalized_section is None:
            return (self._Iy + self._Iz) / max(self._A, 1.0e-30)

        value = self.cross_section.get("geometric_polar_radius_squared")
        if value is None:
            return 0.0
        try:
            polar_radius_squared = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cross_section['geometric_polar_radius_squared'] must be a "
                "finite non-negative value in m^2 for a generalized beam "
                "section."
            ) from exc
        if (
            not np.isfinite(polar_radius_squared)
            or polar_radius_squared < 0.0
        ):
            raise ValueError(
                "cross_section['geometric_polar_radius_squared'] must be a "
                "finite non-negative value in m^2 for a generalized beam "
                "section."
            )
        return polar_radius_squared

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((2, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Beam element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        return coords+getattr(self,'_linear_nodal_offsets',0)

    def _beam_frame_and_transform(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        length = float(np.linalg.norm(coords[1] - coords[0]))
        if length < _SMALL:
            raise ValueError(f"Beam element {self.element_id} has near-zero length")
        e1 = (coords[1] - coords[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for i in range(2):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T@self._linear_rigid_arm_map()

    def _local_linear_stiffness(self, length: float, material: "Material", include_axial: bool = True) -> np.ndarray:
        L = float(length)
        if self.generalized_section is not None:
            section_stiffness = generalized_beam_stiffness(
                self.generalized_section
            )
            basic_transform = _beam2_basic_deformation_matrix(L)
            if not include_axial:
                basic_transform = basic_transform.copy()
                basic_transform[0, :] = 0.0
            _flexibility, basic_stiffness = (
                _beam2_generalized_flexibility(L, section_stiffness)
            )
            # Force-based integration satisfies beam equilibrium exactly and
            # avoids shear locking.  For a diagonal section matrix this
            # reduces to the historical closed-form Timoshenko stiffness;
            # off-diagonal terms retain all generalized couplings.
            return (
                basic_transform.T
                @ basic_stiffness
                @ basic_transform
            )
        E, G12, G13, GJ = _beam_material_properties(material, self.cross_section)
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        K = np.zeros((12, 12), dtype=float)
        if include_axial:
            K[0, 0] = K[6, 6] = EA / L
            K[0, 6] = K[6, 0] = -EA / L

        # Bending about local y: deflection w (local z), rotation ry, EIy,
        # Timoshenko shear parameter from the local-z shear area.
        phi_w = 12.0 * EIy / max(G13 * self._A * self._kz * L**2, _SMALL)
        K[2, 2] = K[8, 8] = 12.0 * EIy / (L**3 * (1.0 + phi_w))
        K[2, 8] = K[8, 2] = -K[2, 2]
        K[2, 4] = K[4, 2] = -6.0 * EIy / (L**2 * (1.0 + phi_w))
        K[2, 10] = K[10, 2] = K[2, 4]
        K[8, 4] = K[4, 8] = -K[2, 4]
        K[8, 10] = K[10, 8] = -K[2, 4]
        K[4, 4] = K[10, 10] = (4.0 + phi_w) * EIy / (L * (1.0 + phi_w))
        K[4, 10] = K[10, 4] = (2.0 - phi_w) * EIy / (L * (1.0 + phi_w))

        # Bending about local z: deflection v (local y), rotation rz, EIz,
        # Timoshenko shear parameter from the local-y shear area.
        phi_v = 12.0 * EIz / max(G12 * self._A * self._ky * L**2, _SMALL)
        K[1, 1] = K[7, 7] = 12.0 * EIz / (L**3 * (1.0 + phi_v))
        K[1, 7] = K[7, 1] = -K[1, 1]
        K[1, 5] = K[5, 1] = 6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[1, 11] = K[11, 1] = 6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[7, 5] = K[5, 7] = -6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[7, 11] = K[11, 7] = -6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[5, 5] = K[11, 11] = (4.0 + phi_v) * EIz / (L * (1.0 + phi_v))
        K[5, 11] = K[11, 5] = (2.0 - phi_v) * EIz / (L * (1.0 + phi_v))

        K[3, 3] = K[9, 9] = GJ / L
        K[3, 9] = K[9, 3] = -GJ / L
        return K

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            if np.any(getattr(self,'_linear_nodal_offsets',0)):
                raise
            return np.zeros((self.total_dofs, self.total_dofs))
        K = self._local_linear_stiffness(L, material)
        K_global = T.T @ K @ T
        self._stiffness_matrix = K_global
        return K_global

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        self._refuse_offset_route('mass/dynamics')
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs))
        generalized_mass = (
            None
            if self.generalized_section is None
            else generalized_beam_mass_matrix(self.generalized_section)
        )
        if generalized_mass is not None:
            # Exact consistent integration of the linear interpolation.  A
            # full sectional inertia matrix may include centre-of-mass or
            # product-of-inertia translation/rotation coupling, so diagonal
            # lumping would discard part of the supplied constitutive data.
            interpolation_mass = np.array(
                [[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]],
                dtype=float,
            )
            M = np.zeros((12, 12), dtype=float)
            for i in range(2):
                for j in range(2):
                    M[
                        i * 6 : (i + 1) * 6,
                        j * 6 : (j + 1) * 6,
                    ] = interpolation_mass[i, j] * L * generalized_mass
            M_global = T.T @ M @ T
            self._mass_matrix = M_global
            return M_global
        rho = material.density
        M = np.zeros((12, 12), dtype=float)
        if bool(self.cross_section.get("consistent_mass", False)):
            # Consistent mass on the element's own linear interpolation,
            # mirroring the 3-node quadratic beam: N_i N_j blocks for the
            # translations (rho A) and for each rotation with its matching
            # section inertia (polar Iy+Iz for torsion).  Rigid-body
            # translational and rotational inertia are exact, so no lumped
            # bar-length correction terms are needed.
            coupling = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=float)
            rotary_density = (
                rho * (self._Iy + self._Iz),  # rx (torsion)
                rho * self._Iy,  # ry
                rho * self._Iz,  # rz
            )
            for i in range(2):
                for j in range(2):
                    factor = coupling[i, j] * L
                    for d in range(3):
                        M[i * 6 + d, j * 6 + d] = rho * self._A * factor
                        M[i * 6 + 3 + d, j * 6 + 3 + d] = rotary_density[d] * factor
            M_global = T.T @ M @ T
            self._mass_matrix = M_global
            return M_global
        mass_per_node = rho * self._A * L / 2.0
        # Lumped rotary inertia: bending rotations carry the rigid-bar term
        # rho*A*L^3/24 plus the section rotatory term rho*I*L/2; the torsion
        # DOF carries the polar section inertia rho*(Iy+Iz)*L/2, not a
        # bar-length term, because spinning about the member axis does not
        # translate the distributed mass.
        rotary_bar = mass_per_node * L**2 / 12.0
        rotary = (
            rho * (self._Iy + self._Iz) * L / 2.0,  # rx (torsion)
            rotary_bar + rho * self._Iy * L / 2.0,  # ry
            rotary_bar + rho * self._Iz * L / 2.0,  # rz
        )
        for i in range(2):
            b = i * 6
            for d in range(3):
                M[b + d, b + d] = mass_per_node
                M[b + 3 + d, b + 3 + d] = rotary[d]
        M_global = T.T @ M @ T
        self._mass_matrix = M_global
        return M_global

    @staticmethod
    def _axial_compression_from_state(state: Optional[Any]) -> float:
        """Return positive compression from a user/reference element state."""
        if state is None:
            return 0.0
        if isinstance(state, (int, float, np.number)):
            return float(state)
        if not isinstance(state, dict):
            return 0.0

        for key in ("axial_compression", "compression", "N_compression"):
            if key in state:
                return float(state[key])

        for key in ("axial_force", "N", "reference_axial_force"):
            if key in state:
                # Stress recovery uses positive axial strain/stress in tension,
                # so a negative axial force is destabilizing compression.
                return -float(state[key])
        return 0.0

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Beam-column geometric stiffness for an axial reference compression.

        The returned matrix follows ``K phi = lambda KG phi``. A positive
        ``axial_compression`` in ``state`` therefore produces a positive
        destabilizing ``KG``.
        """
        self._refuse_offset_route('geometric stiffness')
        axial_compression = self._axial_compression_from_state(state)
        if axial_compression == 0.0:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        factor = axial_compression / (30.0 * L)
        L2 = L * L
        g_standard = factor * np.array(
            [
                [36.0, 3.0 * L, -36.0, 3.0 * L],
                [3.0 * L, 4.0 * L2, -3.0 * L, -L2],
                [-36.0, -3.0 * L, 36.0, -3.0 * L],
                [3.0 * L, -L2, -3.0 * L, 4.0 * L2],
            ],
            dtype=float,
        )

        K_geo = np.zeros((12, 12), dtype=float)
        v_rz = [1, 5, 7, 11]
        w_ry = [2, 4, 8, 10]
        sign_w = np.diag([1.0, -1.0, 1.0, -1.0])
        g_w = sign_w @ g_standard @ sign_w

        for i, row in enumerate(v_rz):
            for j, col in enumerate(v_rz):
                K_geo[row, col] += g_standard[i, j]
        for i, row in enumerate(w_ry):
            for j, col in enumerate(w_ry):
                K_geo[row, col] += g_w[i, j]

        # Wagner term: axial compression destabilizes twist about the shear
        # center, enabling torsional and flexural-torsional column modes
        # (stiffener tripping under axial stress).  The shear center is taken
        # at the centroid and warping is neglected, consistent with the
        # element's St. Venant torsion treatment, so the torsional critical
        # load is G*J*A/Ip.
        polar_ratio = self._geometric_polar_radius_squared()
        g_torsion = axial_compression * polar_ratio / L
        K_geo[3, 3] += g_torsion
        K_geo[3, 9] -= g_torsion
        K_geo[9, 3] -= g_torsion
        K_geo[9, 9] += g_torsion

        return T.T @ K_geo @ T

    def _fiber_plasticity_config(self, material: "Material") -> Optional[FiberSectionPlasticityConfig]:
        config = self._fiber_plasticity
        if config is None:
            return None
        if isinstance(config, dict):
            config = FiberSectionPlasticityConfig(**config)
        elif config is True:
            config = FiberSectionPlasticityConfig()
        elif not isinstance(config, FiberSectionPlasticityConfig):
            raise TypeError("cross_section['fiber_plasticity'] must be a FiberSectionPlasticityConfig, dict or True")
        curve = config.material_curve or getattr(material, "hardening_curve", None)
        if curve is None:
            if (
                _elastic_symmetry(material) == "orthotropic"
                and getattr(material, "hill_yield", None) is not None
            ):
                # Hill X supplies an elastic-perfectly-plastic longitudinal
                # fiber law when no scalar hardening curve is provided.
                return config
            return None
        if config.material_curve is curve:
            return config
        return FiberSectionPlasticityConfig(config.num_y, config.num_z, curve)

    def _fiber_section_grid(self, config: FiberSectionPlasticityConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cross = self.cross_section if isinstance(getattr(self, "cross_section", None), dict) else {}
        web_height = float(cross.get("web_height") or 0.0)
        web_thickness = float(cross.get("web_thickness") or 0.0)
        flange_width = float(cross.get("flange_width") or 0.0)
        flange_thickness = float(cross.get("flange_thickness") or 0.0)
        shaped = (
            web_height > 0.0
            and web_thickness > 0.0
            and flange_width > 0.0
            and flange_thickness > 0.0
        )
        key = (
            "fiber_grid",
            int(config.num_y),
            int(config.num_z),
            float(self._A),
            float(self._Iy),
            float(self._Iz),
            round(web_height, 12),
            round(web_thickness, 12),
            round(flange_width, 12),
            round(flange_thickness, 12),
        )
        cache = getattr(self, "_fiber_grid_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache["y"], cache["z"], cache["weights"]

        if shaped:
            # Profile-true fiber layout: web and flange strips follow the real
            # T/L section geometry so the plastification order (flange tips vs
            # web) is captured.  The coordinates and weights are afterwards
            # recentred and rescaled to reproduce the section constants
            # A/Iy/Iz exactly, keeping the elastic response identical to the
            # section-property idealization.  Flanges are laid out symmetric
            # in y also for L profiles (the section constants carry no product
            # of inertia).
            n_web = max(int(config.num_z), 3)
            n_flange = max(int(config.num_y), 3)
            web_z = (np.arange(n_web, dtype=float) + 0.5) * (web_height / n_web)
            y_parts = [np.zeros(n_web)]
            z_parts = [web_z]
            weight_parts = [np.full(n_web, web_height * web_thickness / n_web)]
            flange_y = (np.arange(n_flange, dtype=float) + 0.5) * (flange_width / n_flange) - 0.5 * flange_width
            for z_level in (web_height + 0.25 * flange_thickness, web_height + 0.75 * flange_thickness):
                y_parts.append(flange_y.copy())
                z_parts.append(np.full(n_flange, z_level))
                weight_parts.append(np.full(n_flange, flange_width * flange_thickness / (2 * n_flange)))
            y = np.concatenate(y_parts)
            z = np.concatenate(z_parts)
            weights = np.concatenate(weight_parts)
            total = float(np.sum(weights))
            if total > _SMALL and self._A > 0.0:
                weights *= float(self._A) / total
            area = float(np.sum(weights))
            if area > _SMALL:
                y = y - float(np.sum(weights * y)) / area
                z = z - float(np.sum(weights * z)) / area
        else:
            raw_y = np.linspace(-1.0, 1.0, int(config.num_y)) if config.num_y > 1 else np.zeros(1)
            raw_z = np.linspace(-1.0, 1.0, int(config.num_z)) if config.num_z > 1 else np.zeros(1)
            yy, zz = np.meshgrid(raw_y, raw_z, indexing="ij")
            y = yy.reshape(-1)
            z = zz.reshape(-1)
            weights = np.full(y.size, float(self._A) / max(y.size, 1), dtype=float)

        denom_y = float(np.sum(weights * y * y))
        denom_z = float(np.sum(weights * z * z))
        if denom_y > _SMALL and self._Iz > 0.0:
            y *= np.sqrt(float(self._Iz) / denom_y)
        else:
            y *= 0.0
        if denom_z > _SMALL and self._Iy > 0.0:
            z *= np.sqrt(float(self._Iy) / denom_z)
        else:
            z *= 0.0

        self._fiber_grid_cache = {"key": key, "y": y, "z": z, "weights": weights}
        return y, z, weights

    @staticmethod
    def _uniaxial_return_map(
        strain: np.ndarray,
        state: Optional[Any],
        E: float,
        curve: Any,
        flow_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        strain = np.asarray(strain, dtype=float).reshape(-1)
        n = strain.size
        if isinstance(state, dict) and np.asarray(state.get("plastic_strain", [])).size == n:
            plastic_old = np.asarray(state["plastic_strain"], dtype=float).reshape(-1)
            alpha_old = np.asarray(state.get("alpha", np.zeros(n)), dtype=float).reshape(-1)
            if alpha_old.size == 1:
                alpha_old = np.full(n, float(alpha_old[0]), dtype=float)
            elif alpha_old.size != n:
                raise ValueError(
                    f"Beam fiber alpha has {alpha_old.size} entries; expected 1 or {n}"
                )
        else:
            plastic_old = np.zeros(n, dtype=float)
            alpha_old = np.zeros(n, dtype=float)

        trial = E * (strain - plastic_old)
        abs_trial = np.abs(trial)
        scale = float(flow_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("beam fiber flow_scale must be positive")
        flow_old = (
            np.full(n, scale, dtype=float)
            if curve is None
            else scale * curve.flow_stress(alpha_old)
        )
        yielding = abs_trial > flow_old + 1.0e-9 * np.maximum(flow_old, 1.0)

        stress = trial.copy()
        tangent = np.full(n, E, dtype=float)
        plastic_new = plastic_old.copy()
        alpha_new = alpha_old.copy()
        if not np.any(yielding):
            return stress, tangent, plastic_new, alpha_new

        indices = np.where(yielding)[0]
        for idx in indices:
            sign = 1.0 if trial[idx] >= 0.0 else -1.0
            dgamma = 0.0
            lower = 0.0
            upper = abs_trial[idx] / E
            H = (
                0.0
                if curve is None
                else scale
                * float(
                    curve.hardening_modulus(np.array([alpha_old[idx]]))[0]
                )
            )
            converged = False
            for _ in range(30):
                alpha_trial = alpha_old[idx] + dgamma
                if curve is None:
                    sy = scale
                    H = 0.0
                else:
                    sy = scale * float(
                        curve.flow_stress(np.array([alpha_trial]))[0]
                    )
                    H = scale * float(
                        curve.hardening_modulus(np.array([alpha_trial]))[0]
                    )
                residual = abs_trial[idx] - E * dgamma - sy
                # At very large trial strains, ``abs_trial - E * dgamma``
                # loses several hundred pascals to floating-point cancellation.
                # Do not demand a residual below the resolution of the terms
                # being subtracted.  The physical tolerance remains governing
                # for ordinary structural strain ranges.
                roundoff_tolerance = 16.0 * np.finfo(float).eps * max(
                    abs_trial[idx], abs(E * dgamma), abs(sy), 1.0
                )
                convergence_tolerance = max(
                    1.0e-8 * max(abs(sy), 1.0), roundoff_tolerance
                )
                # Yield entry is tighter than the closure tolerance. A point
                # admitted as yielding must take a positive return update
                # before reporting a plastic tangent; otherwise its unchanged
                # elastic stress has derivative E, not E*H/(E+H).
                if dgamma > 0.0 and abs(residual) <= convergence_tolerance:
                    converged = True
                    break
                if residual > 0.0:
                    lower = dgamma
                else:
                    upper = dgamma
                newton = dgamma + residual / max(E + H, _SMALL)
                if (
                    not np.isfinite(newton)
                    or newton <= lower
                    or newton >= upper
                ):
                    newton = 0.5 * (lower + upper)
                dgamma = newton
            if not converged:
                raise RuntimeError(
                    "Beam fiber return mapping did not converge after 30 iterations"
                )
            stress[idx] = sign * max(abs_trial[idx] - E * dgamma, 0.0)
            plastic_new[idx] = plastic_old[idx] + sign * dgamma
            alpha_new[idx] = alpha_old[idx] + dgamma
            tangent[idx] = E * H / max(E + H, _SMALL)
        return stress, tangent, plastic_new, alpha_new

    def _compute_fiber_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any],
        config: FiberSectionPlasticityConfig,
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        coords = self.get_node_coordinates(mesh)
        L, T = self._beam_frame_and_transform(coords)
        u_loc = T @ np.asarray(u_elem, dtype=float)
        E, G12, G13, GJ = _beam_material_properties(material, self.cross_section)
        y, z, weights = self._fiber_section_grid(config)

        du = u_loc[6] - u_loc[0]
        dv = u_loc[7] - u_loc[1]
        dw = u_loc[8] - u_loc[2]
        eps0 = du / L + (dv**2 + dw**2) / (2.0 * L**2)
        kappa_y = (u_loc[10] - u_loc[4]) / L
        kappa_z = (u_loc[11] - u_loc[5]) / L
        fiber_strain = eps0 + z * kappa_y + y * kappa_z
        initial_state = _copy_state_fields(state, _INITIAL_BEAM_STATE_KEYS)
        initial_stress = _beam_fiber_field(
            initial_state.get("initial_fiber_stress", 0.0),
            fiber_strain.size,
            fiber_strain.size,
            "initial_fiber_stress",
        )
        initial_prestrain = _beam_fiber_field(
            initial_state.get("initial_fiber_prestrain", 0.0),
            fiber_strain.size,
            fiber_strain.size,
            "initial_fiber_prestrain",
        )
        constitutive_fiber_strain = fiber_strain - initial_prestrain + initial_stress / E
        stress, Et, plastic_new, alpha_new = self._uniaxial_return_map(
            constitutive_fiber_strain,
            state,
            E,
            config.material_curve,
            _beam_flow_scale(material, config.material_curve),
        )

        B = np.zeros((fiber_strain.size, 12), dtype=float)
        B[:, 0] = -1.0 / L
        B[:, 6] = 1.0 / L
        B[:, 1] = -dv / L**2
        B[:, 7] = dv / L**2
        B[:, 2] = -dw / L**2
        B[:, 8] = dw / L**2
        B[:, 4] = -z / L
        B[:, 10] = z / L
        B[:, 5] = -y / L
        B[:, 11] = y / L

        F_loc = L * np.einsum("i,i,ij->j", weights, stress, B)
        K_loc = None
        if tangent:
            K_loc = L * np.einsum("i,i,ij,ik->jk", weights, Et, B, B)

        N_force = float(np.sum(weights * stress))
        if tangent:
            string = N_force / L
            for a, b in ((1, 7), (2, 8)):
                K_loc[a, a] += string
                K_loc[b, b] += string
                K_loc[a, b] -= string
                K_loc[b, a] -= string

        B_shear_y = np.zeros(12, dtype=float)
        B_shear_y[1], B_shear_y[7] = -1.0 / L, 1.0 / L
        B_shear_y[5], B_shear_y[11] = -0.5, -0.5
        B_shear_z = np.zeros(12, dtype=float)
        B_shear_z[2], B_shear_z[8] = -1.0 / L, 1.0 / L
        B_shear_z[4], B_shear_z[10] = 0.5, 0.5
        B_torsion = np.zeros(12, dtype=float)
        B_torsion[3], B_torsion[9] = -1.0 / L, 1.0 / L
        # The one-section fiber state resolves constant curvature.  The
        # remaining elastic curvature/shear compliance must be condensed just
        # as in _local_linear_stiffness: 1/S_eff = 1/S + L^2/(12*EI).
        # Using the uncondensed kGA here made merely enabling an unyielded
        # fiber law change the elastic Timoshenko response (shear locking).
        # This retains the existing one-section plastic approximation, fiber
        # strains, return map and history; it adds no distributed yield model.
        phi_y = 12.0 * E * self._Iz / max(G12 * self._A * self._ky * L**2, _SMALL)
        phi_z = 12.0 * E * self._Iy / max(G13 * self._A * self._kz * L**2, _SMALL)
        shear_y = 12.0 * E * self._Iz / (L**2 * (1.0 + phi_y))
        shear_z = 12.0 * E * self._Iy / (L**2 * (1.0 + phi_z))
        K_aux = L * (
            shear_y * np.outer(B_shear_y, B_shear_y)
            + shear_z * np.outer(B_shear_z, B_shear_z)
            + GJ * np.outer(B_torsion, B_torsion)
        )
        F_loc += K_aux @ u_loc
        if tangent:
            K_loc += K_aux

        trial_state = {
            "plastic_strain": plastic_new,
            "alpha": alpha_new,
            "fiber_strain": constitutive_fiber_strain.copy(),
            "kinematic_fiber_strain": fiber_strain.copy(),
            "fiber_stress": stress.copy(),
            "axial_force": N_force,
            "equivalent_stress_measure": (
                "hill48" if _elastic_symmetry(material) == "orthotropic" else "von_mises"
            ),
            **initial_state,
        }
        if not tangent:
            return T.T @ F_loc, None, trial_state
        return T.T @ F_loc, T.T @ K_loc @ T, trial_state

    def _current_beam_frame_and_transform(self, coords: np.ndarray, u_elem: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        current = coords.copy()
        current[0] += np.asarray(u_elem[0:3], dtype=float)
        current[1] += np.asarray(u_elem[6:9], dtype=float)
        length = float(np.linalg.norm(current[1] - current[0]))
        if length < _SMALL:
            raise ValueError(f"Beam element {self.element_id} has near-zero current length")
        e1 = (current[1] - current[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for i in range(2):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T, R, current

    def _compute_corotational_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any],
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Elastic 2-node corotational beam response.

        The element frame follows the current chord.  Rigid-body chord rotation
        is subtracted from nodal rotations, so a finite rigid rotation with
        matching nodal rotation vectors produces near-zero internal force.
        """
        coords = self.get_node_coordinates(mesh)
        L0, _T0 = self._beam_frame_and_transform(coords)
        Lc, Tc, Rc, _current = self._current_beam_frame_and_transform(coords, np.asarray(u_elem, dtype=float))
        R0 = _beam_rotation_matrix((coords[1] - coords[0]) / L0, self._orientation)
        rigid_rotation = _rotation_vector_from_matrix(Rc @ R0.T)

        u_global = np.asarray(u_elem, dtype=float).reshape(12)
        q = np.zeros(12, dtype=float)
        q[6] = Lc - L0
        q[3:6] = Rc.T @ (u_global[3:6] - rigid_rotation)
        q[9:12] = Rc.T @ (u_global[9:12] - rigid_rotation)

        K_loc = self._local_linear_stiffness(L0, material)
        F_loc = K_loc @ q
        F_global = Tc.T @ F_loc
        trial_state = {
            "geometric_nonlinearity": "corotational",
            "initial_length": float(L0),
            "current_length": float(Lc),
            "axial_extension": float(Lc - L0),
            "rigid_rotation_vector": rigid_rotation.copy(),
            "basic_deformation_norm": float(np.linalg.norm(q)),
        }
        if self.generalized_section is not None:
            section_stiffness = generalized_beam_stiffness(
                self.generalized_section
            )
            basic_deformation = (
                _beam2_basic_deformation_matrix(L0) @ q
            )
            _flexibility, basic_stiffness = (
                _beam2_generalized_flexibility(
                    L0,
                    section_stiffness,
                )
            )
            basic_force = basic_stiffness @ basic_deformation
            stations = np.array((-1.0, 0.0, 1.0), dtype=float)
            generalized_resultant = np.asarray(
                [
                    _beam2_force_interpolation(L0, xi) @ basic_force
                    for xi in stations
                ],
                dtype=float,
            )
            generalized_strain = (
                generalized_resultant
                @ np.linalg.inv(section_stiffness).T
            )
            trial_state.update(
                {
                    "generalized_section": True,
                    "generalized_strain": generalized_strain.copy(),
                    "generalized_resultant": generalized_resultant.copy(),
                    "generalized_station_coordinates": stations,
                    "generalized_strain_order": GENERALIZED_BEAM_STRAIN_ORDER,
                    "generalized_resultant_order": (
                        GENERALIZED_BEAM_RESULTANT_ORDER
                    ),
                    "axial_force": float(
                        np.mean(generalized_resultant[:, 0])
                    ),
                    "recovery_scope": "section_resultants_only",
                }
            )
        if not tangent:
            return F_global, None, trial_state
        K_global = Tc.T @ K_loc @ Tc
        return F_global, K_global, trial_state

    def _compute_generalized_von_karman_response(
        self,
        mesh: "FEMesh",
        u_elem: np.ndarray,
        state: Optional[Any],
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Coupled generalized-section response from one strain-energy law."""

        coords = self.get_node_coordinates(mesh)
        L, T = self._beam_frame_and_transform(coords)
        u_loc = T @ np.asarray(u_elem, dtype=float)
        basic_transform = _beam2_basic_deformation_matrix(L)
        basic_deformation = basic_transform @ u_loc
        dv = float(u_loc[7] - u_loc[1])
        dw = float(u_loc[8] - u_loc[2])
        basic_deformation[0] += (dv**2 + dw**2) / (2.0 * L)
        basic_tangent = basic_transform.copy()
        basic_tangent[0, 1], basic_tangent[0, 7] = -dv / L, dv / L
        basic_tangent[0, 2], basic_tangent[0, 8] = -dw / L, dw / L

        section_stiffness = generalized_beam_stiffness(
            self.generalized_section
        )
        _flexibility, basic_stiffness = _beam2_generalized_flexibility(
            L,
            section_stiffness,
        )
        basic_force = basic_stiffness @ basic_deformation
        F_loc = basic_tangent.T @ basic_force
        stations = np.array((-1.0, 0.0, 1.0), dtype=float)
        generalized_resultant = np.asarray(
            [
                _beam2_force_interpolation(L, xi) @ basic_force
                for xi in stations
            ],
            dtype=float,
        )
        generalized_strain = (
            generalized_resultant @ np.linalg.inv(section_stiffness).T
        )
        trial_state = {
            **(dict(state) if isinstance(state, dict) else {}),
            "generalized_section": True,
            "geometric_nonlinearity": "von_karman",
            "generalized_strain": generalized_strain.copy(),
            "generalized_resultant": generalized_resultant.copy(),
            "generalized_station_coordinates": stations,
            "generalized_strain_order": GENERALIZED_BEAM_STRAIN_ORDER,
            "generalized_resultant_order": GENERALIZED_BEAM_RESULTANT_ORDER,
            "axial_force": float(np.mean(generalized_resultant[:, 0])),
            "recovery_scope": "section_resultants_only",
        }
        if not tangent:
            return T.T @ F_loc, None, trial_state

        K_loc = basic_tangent.T @ basic_stiffness @ basic_tangent
        # Hessian of the von Karman axial extension, multiplied by N.
        string = float(basic_force[0]) / L
        for a, b in ((1, 7), (2, 8)):
            K_loc[a, a] += string
            K_loc[b, b] += string
            K_loc[a, b] -= string
            K_loc[b, a] -= string
        return T.T @ F_loc, T.T @ K_loc @ T, trial_state

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Elastic beam-column response with von Karman axial coupling.

        The axial strain includes the transverse-displacement rotation terms

            eps = (u2-u1)/L + ((v2-v1)^2 + (w2-w1)^2) / (2 L^2)

        which gives the P-delta string effect consistently (internal force
        and tangent from the same potential).  Bending, shear and torsion
        remain linear elastic unless ``cross_section["fiber_plasticity"]`` is
        provided, in which case axial/bending response is integrated over a
        uniaxial fiber section using the material hardening curve.
        """
        self._refuse_offset_route('nonlinear response')
        if self._geometric_nonlinearity in {"corotational", "co_rotational", "corot"}:
            return self._compute_corotational_nonlinear_response(
                mesh, material, u_elem, state, tangent
            )

        if self.generalized_section is not None:
            return self._compute_generalized_von_karman_response(
                mesh,
                u_elem,
                state,
                tangent,
            )

        fiber_config = self._fiber_plasticity_config(material)
        if fiber_config is not None:
            return self._compute_fiber_nonlinear_response(
                mesh, material, u_elem, state, fiber_config, tangent
            )

        cache = getattr(self, "_nl_cache", None)
        if cache is None:
            coords = self.get_node_coordinates(mesh)
            L, T = self._beam_frame_and_transform(coords)
            K_global = self.compute_stiffness_matrix(mesh, material)
            K_loc = T @ K_global @ T.T
            # Remove the linear axial block; the von Karman axial response
            # replaces it entirely.
            for i in (0, 6):
                K_loc[i, :] = 0.0
                K_loc[:, i] = 0.0
            E1, _G12, _G13, _GJ = _beam_material_properties(
                material,
                self.cross_section,
            )
            cache = {"L": L, "T": T, "K_noax": K_loc, "EA": E1 * self._A}
            self._nl_cache = cache

        L = cache["L"]
        T = cache["T"]
        EA = cache["EA"]
        u_loc = T @ np.asarray(u_elem, dtype=float)

        du = u_loc[6] - u_loc[0]
        dv = u_loc[7] - u_loc[1]
        dw = u_loc[8] - u_loc[2]
        eps = du / L + (dv**2 + dw**2) / (2.0 * L**2)
        N_force = EA * eps

        d_eps = np.zeros(12, dtype=float)
        d_eps[0], d_eps[6] = -1.0 / L, 1.0 / L
        d_eps[1], d_eps[7] = -dv / L**2, dv / L**2
        d_eps[2], d_eps[8] = -dw / L**2, dw / L**2

        F_loc = cache["K_noax"] @ u_loc + N_force * L * d_eps
        if not tangent:
            return T.T @ F_loc, None, state
        K_loc = cache["K_noax"] + EA * L * np.outer(d_eps, d_eps)
        string = N_force / L
        for a, b in ((1, 7), (2, 8)):
            K_loc[a, a] += string
            K_loc[b, b] += string
            K_loc[a, b] -= string
            K_loc[b, a] -= string

        return T.T @ F_loc, T.T @ K_loc @ T, state

    def _end_displacements(self, u_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Local DOF vectors of the two geometric end nodes."""
        return u_local[0:6], u_local[6:12]

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, Any]:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return {}
        u_local = T @ self._get_element_displacements(mesh, displacements)
        if self.generalized_section is not None:
            section_stiffness = generalized_beam_stiffness(
                self.generalized_section
            )
            if isinstance(self, QuadraticBeamElement):
                strain_rows = []
                for xi in self.GAUSS_POINTS:
                    shape, derivative_xi = self.compute_shape_functions(
                        float(xi)
                    )
                    B = _beam_generalized_strain_matrix_from_shape(
                        shape,
                        derivative_xi * 2.0 / L,
                    )
                    strain_rows.append(B @ u_local)
                generalized_strain = np.asarray(strain_rows, dtype=float)
                generalized_resultant = (
                    generalized_strain @ section_stiffness.T
                )
                return _generalized_beam_recovery(
                    generalized_strain,
                    generalized_resultant,
                    stations=np.asarray(self.GAUSS_POINTS, dtype=float),
                )
            basic_transform = _beam2_basic_deformation_matrix(L)
            basic_deformation = basic_transform @ u_local
            _flexibility, basic_stiffness = (
                _beam2_generalized_flexibility(L, section_stiffness)
            )
            basic_force = basic_stiffness @ basic_deformation
            compliance = np.linalg.inv(section_stiffness)
            stations = np.array((-1.0, 0.0, 1.0), dtype=float)
            generalized_resultant = np.asarray(
                [
                    _beam2_force_interpolation(L, xi) @ basic_force
                    for xi in stations
                ],
                dtype=float,
            )
            generalized_strain = generalized_resultant @ compliance.T
            return _generalized_beam_recovery(
                generalized_strain,
                generalized_resultant,
                stations=stations,
            )
        E, G12, G13, GJ = _beam_material_properties(material, self.cross_section)
        end_a, end_b = self._end_displacements(u_local)
        u1, v1, w1, rx1, ry1, rz1 = end_a
        u2, v2, w2, rx2, ry2, rz2 = end_b
        sigma_axial = E * (u2 - u1) / L
        kappa_z = (rz2 - rz1) / L
        kappa_y = (ry2 - ry1) / L
        M_z = E * self._Iz * kappa_z
        M_y = E * self._Iy * kappa_y
        c_y, c_z = self._fiber_distances()
        sigma_bending_y = M_y * c_z / max(self._Iy, _SMALL)
        sigma_bending_z = M_z * c_y / max(self._Iz, _SMALL)
        gamma_xy = (v2 - v1) / L - 0.5 * (rz1 + rz2)
        gamma_xz = (w2 - w1) / L + 0.5 * (ry1 + ry2)
        tau_y = G12 * self._ky * gamma_xy
        tau_z = G13 * self._kz * gamma_xz
        tau_torsion = GJ * (rx2 - rx1) / L / self._torsion_section_modulus()
        sigma_x = sigma_axial + sigma_bending_y + sigma_bending_z
        von_mises = np.sqrt(sigma_x**2 + 3.0 * (tau_y**2 + tau_z**2 + tau_torsion**2))
        recovered = {
            "axial_stress": sigma_axial,
            "bending_stress_y": sigma_bending_y,
            "bending_stress_z": sigma_bending_z,
            "shear_stress_y": tau_y,
            "shear_stress_z": tau_z,
            "torsional_stress": tau_torsion,
            "von_mises": von_mises,
        }
        hill_yield = getattr(material, "hill_yield", None)
        if hill_yield is not None:
            from .plasticity import hill48_equivalent_stress

            # Section recovery supplies envelopes rather than a resolved
            # perimeter stress point.  Include both directional transverse
            # shears, and conservatively combine the torsional envelope with
            # either x-y or x-z shear.
            candidates = np.array(
                [
                    [sigma_x, 0.0, 0.0, 0.0, tau_z, tau_y],
                    [sigma_x, 0.0, 0.0, 0.0, tau_z, tau_y + tau_torsion],
                    [sigma_x, 0.0, 0.0, 0.0, tau_z, tau_y - tau_torsion],
                    [sigma_x, 0.0, 0.0, 0.0, tau_z + tau_torsion, tau_y],
                    [sigma_x, 0.0, 0.0, 0.0, tau_z - tau_torsion, tau_y],
                ],
                dtype=float,
            )
            equivalent = float(
                np.max(hill48_equivalent_stress(candidates, hill_yield))
            )
            recovered["equivalent_stress"] = equivalent
            recovered["equivalent_stress_measure"] = "hill48"
            recovered["hill_utilization"] = equivalent / max(float(hill_yield.X), _SMALL)
            recovered["equivalent_stress_scope"] = (
                "conservative beam-local axial/shear/torsion envelope"
            )
        else:
            recovered["equivalent_stress"] = von_mises
            recovered["equivalent_stress_measure"] = "von_mises"
        return recovered


class QuadraticBeamElement(BeamElement):
    """3-node quadratic Timoshenko beam element with 6 DOF per node."""

    GAUSS_POINTS = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
    GAUSS_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        cross_section: Optional[Dict[str, Any]] = None,
        eccentricity: Optional[np.ndarray] = None,
        section: Any = None,
    ):
        Element.__init__(self, element_id, node_ids, material_name)
        if len(node_ids) != 3:
            raise ValueError(f"QuadraticBeamElement requires 3 nodes, got {len(node_ids)}")
        self.cross_section = cross_section or {}
        self._A = self.cross_section.get("area", 0.01)
        self._Iy = self.cross_section.get("Iy", 1.0e-8)
        self._Iz = self.cross_section.get("Iz", 1.0e-8)
        self._J = self.cross_section.get("J", 1.0e-8)
        self._ky = self.cross_section.get("shear_factor_y", 5.0 / 6.0)
        self._kz = self.cross_section.get("shear_factor_z", 5.0 / 6.0)
        self._orientation = _section_orientation(self.cross_section)
        self._c_y = self.cross_section.get("c_y")
        self._c_z = self.cross_section.get("c_z")
        self._torsion_modulus = self.cross_section.get("torsion_modulus")
        self._fiber_plasticity = self.cross_section.get("fiber_plasticity")
        self._geometric_nonlinearity = str(
            self.cross_section.get(
                "geometric_nonlinearity",
                self.cross_section.get("geometry", "von_karman"),
            )
        ).lower()
        self.generalized_section = resolve_generalized_beam_section(
            self.cross_section,
            section,
        )
        if self.generalized_section is not None and self._fiber_plasticity is not None:
            raise ValueError(
                "Generalized beam sections do not support "
                "cross_section['fiber_plasticity']; supply an elastic section "
                "without fiber plasticity."
            )
        if self.generalized_section is not None:
            self._geometric_polar_radius_squared()
        if self.generalized_section is not None and self._geometric_nonlinearity in {
            "corotational",
            "co_rotational",
            "corot",
        }:
            raise ValueError(
                "QuadraticBeamElement does not support corotational kinematics "
                "with a generalized beam section; use BeamElement or "
                "geometric_nonlinearity='von_karman'."
            )
        self.eccentricity = np.zeros(3, dtype=float) if eccentricity is None else np.asarray(eccentricity, dtype=float)

    def _end_displacements(self, u_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Geometric end nodes are 0 and 2; node 1 is the midside node.  The
        # end-difference over the full length equals the quadratic shape
        # function derivative evaluated at the element centre.
        return u_local[0:6], u_local[12:18]

    def _compute_generalized_von_karman_response_quadratic(
        self,
        length: float,
        transform: np.ndarray,
        u_local: np.ndarray,
        state: Optional[Any],
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Quadratic coupled-section response and exact energy tangent."""

        L = float(length)
        section_stiffness = generalized_beam_stiffness(
            self.generalized_section
        )
        F_loc = np.zeros(18, dtype=float)
        K_loc = np.zeros((18, 18), dtype=float) if tangent else None
        strain_rows = []
        resultant_rows = []
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            shape, derivative_xi = self.compute_shape_functions(float(xi))
            derivative_x = derivative_xi * 2.0 / L
            B_linear = _beam_generalized_strain_matrix_from_shape(
                shape,
                derivative_x,
            )
            B_v = np.zeros(18, dtype=float)
            B_w = np.zeros(18, dtype=float)
            for i in range(3):
                B_v[i * 6 + 1] = derivative_x[i]
                B_w[i * 6 + 2] = derivative_x[i]
            v_gradient = float(B_v @ u_local)
            w_gradient = float(B_w @ u_local)

            generalized_strain = B_linear @ u_local
            generalized_strain[0] += 0.5 * (
                v_gradient**2 + w_gradient**2
            )
            B_tangent = B_linear.copy()
            B_tangent[0, :] += (
                v_gradient * B_v + w_gradient * B_w
            )
            generalized_resultant = (
                section_stiffness @ generalized_strain
            )
            jacobian_weight = L / 2.0 * float(weight)
            F_loc += (
                jacobian_weight
                * (B_tangent.T @ generalized_resultant)
            )
            if tangent:
                K_loc += jacobian_weight * (
                    B_tangent.T @ section_stiffness @ B_tangent
                    + float(generalized_resultant[0])
                    * (np.outer(B_v, B_v) + np.outer(B_w, B_w))
                )
            strain_rows.append(generalized_strain.copy())
            resultant_rows.append(generalized_resultant.copy())

        generalized_strains = np.asarray(strain_rows, dtype=float)
        generalized_resultants = np.asarray(resultant_rows, dtype=float)
        trial_state = {
            **(dict(state) if isinstance(state, dict) else {}),
            "generalized_section": True,
            "geometric_nonlinearity": "von_karman",
            "generalized_strain": generalized_strains,
            "generalized_resultant": generalized_resultants,
            "generalized_station_coordinates": np.asarray(
                self.GAUSS_POINTS,
                dtype=float,
            ).copy(),
            "generalized_strain_order": GENERALIZED_BEAM_STRAIN_ORDER,
            "generalized_resultant_order": GENERALIZED_BEAM_RESULTANT_ORDER,
            "axial_force": float(
                np.mean(generalized_resultants[:, 0])
            ),
            "recovery_scope": "section_resultants_only",
        }
        if not tangent:
            return transform.T @ F_loc, None, trial_state
        return (
            transform.T @ F_loc,
            transform.T @ K_loc @ transform,
            trial_state,
        )

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Gauss-point von Karman beam-column response on the quadratic interpolation.

        At each integration point the axial strain includes the transverse
        gradient coupling

            eps = u' + (v'^2 + w'^2) / 2

        with all gradients from the quadratic shape functions, giving internal
        force and consistent tangent (including the string/geometric term) from
        the same potential.  With ``cross_section["fiber_plasticity"]`` the
        axial/bending response is integrated over the uniaxial fiber grid per
        Gauss point using the material hardening curve; shear and torsion stay
        linear elastic, matching the 2-node beam formulation.
        """
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return Element.compute_nonlinear_response(
                self, mesh, material, u_elem, state, num_layers, tangent
            )
        u_loc = T @ np.asarray(u_elem, dtype=float)
        if self.generalized_section is not None:
            return self._compute_generalized_von_karman_response_quadratic(
                L,
                T,
                u_loc,
                state,
                tangent,
            )
        E, G12, G13, GJ = _beam_material_properties(material, self.cross_section)
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GA_y = G12 * self._A * self._ky
        GA_z = G13 * self._A * self._kz

        fiber_config = self._fiber_plasticity_config(material)
        num_gp = len(self.GAUSS_POINTS)
        if fiber_config is not None:
            y_f, z_f, w_f = self._fiber_section_grid(fiber_config)
            n_fibers = y_f.size
            fiber_strain_all = np.zeros(num_gp * n_fibers, dtype=float)
            B_fiber_all = np.zeros((num_gp * n_fibers, 18), dtype=float)

        F_loc = np.zeros(18, dtype=float)
        K_loc = np.zeros((18, 18), dtype=float) if tangent else None
        gp_data = []
        axial_forces = []
        for gp_index, (xi, weight) in enumerate(zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS)):
            N, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            jac = L / 2.0 * float(weight)

            B_ax = np.zeros(18, dtype=float)
            B_v = np.zeros(18, dtype=float)
            B_w = np.zeros(18, dtype=float)
            B_ry = np.zeros(18, dtype=float)
            B_rz = np.zeros(18, dtype=float)
            B_torsion = np.zeros(18, dtype=float)
            B_shear_xy = np.zeros(18, dtype=float)
            B_shear_xz = np.zeros(18, dtype=float)
            for i in range(3):
                b = i * 6
                B_ax[b + 0] = dN_dx[i]
                B_v[b + 1] = dN_dx[i]
                B_w[b + 2] = dN_dx[i]
                B_torsion[b + 3] = dN_dx[i]
                B_ry[b + 4] = dN_dx[i]
                B_rz[b + 5] = dN_dx[i]
                B_shear_xy[b + 1] = dN_dx[i]
                B_shear_xy[b + 5] = -N[i]
                B_shear_xz[b + 2] = dN_dx[i]
                B_shear_xz[b + 4] = N[i]

            v_grad = float(B_v @ u_loc)
            w_grad = float(B_w @ u_loc)
            eps0 = float(B_ax @ u_loc) + 0.5 * (v_grad**2 + w_grad**2)
            kappa_y = float(B_ry @ u_loc)
            kappa_z = float(B_rz @ u_loc)
            B_membrane = B_ax + v_grad * B_v + w_grad * B_w

            # linear shear and torsion at every integration point
            gamma_xy = float(B_shear_xy @ u_loc)
            gamma_xz = float(B_shear_xz @ u_loc)
            twist = float(B_torsion @ u_loc)
            F_loc += jac * (GA_y * gamma_xy * B_shear_xy + GA_z * gamma_xz * B_shear_xz + GJ * twist * B_torsion)
            if tangent:
                K_loc += jac * (
                    GA_y * np.outer(B_shear_xy, B_shear_xy)
                    + GA_z * np.outer(B_shear_xz, B_shear_xz)
                    + GJ * np.outer(B_torsion, B_torsion)
                )

            if fiber_config is not None:
                rows = slice(gp_index * n_fibers, (gp_index + 1) * n_fibers)
                fiber_strain_all[rows] = eps0 + z_f * kappa_y + y_f * kappa_z
                B_fiber_all[rows] = (
                    B_membrane[None, :] + z_f[:, None] * B_ry[None, :] + y_f[:, None] * B_rz[None, :]
                )
                gp_data.append((jac, B_v, B_w, rows))
            else:
                axial_force = EA * eps0
                axial_forces.append(axial_force)
                F_loc += jac * (axial_force * B_membrane + EIy * kappa_y * B_ry + EIz * kappa_z * B_rz)
                if tangent:
                    K_loc += jac * (
                        EA * np.outer(B_membrane, B_membrane)
                        + axial_force * (np.outer(B_v, B_v) + np.outer(B_w, B_w))
                        + EIy * np.outer(B_ry, B_ry)
                        + EIz * np.outer(B_rz, B_rz)
                    )

        trial_state: Optional[Any] = state
        if fiber_config is not None:
            initial_state = _copy_state_fields(state, _INITIAL_BEAM_STATE_KEYS)
            initial_stress = _beam_fiber_field(
                initial_state.get("initial_fiber_stress", 0.0),
                fiber_strain_all.size,
                n_fibers,
                "initial_fiber_stress",
            )
            initial_prestrain = _beam_fiber_field(
                initial_state.get("initial_fiber_prestrain", 0.0),
                fiber_strain_all.size,
                n_fibers,
                "initial_fiber_prestrain",
            )
            constitutive_fiber_strain = (
                fiber_strain_all - initial_prestrain + initial_stress / E
            )
            stress, Et, plastic_new, alpha_new = self._uniaxial_return_map(
                constitutive_fiber_strain,
                state,
                E,
                fiber_config.material_curve,
                _beam_flow_scale(material, fiber_config.material_curve),
            )
            for jac, B_v, B_w, rows in gp_data:
                weights_gp = w_f
                stress_gp = stress[rows]
                F_loc += jac * np.einsum("f,f,fj->j", weights_gp, stress_gp, B_fiber_all[rows])
                axial_force = float(np.sum(weights_gp * stress_gp))
                axial_forces.append(axial_force)
                if tangent:
                    K_loc += jac * np.einsum("f,f,fj,fk->jk", weights_gp, Et[rows], B_fiber_all[rows], B_fiber_all[rows])
                    K_loc += jac * axial_force * (np.outer(B_v, B_v) + np.outer(B_w, B_w))
            trial_state = {
                "plastic_strain": plastic_new,
                "alpha": alpha_new,
                "fiber_strain": constitutive_fiber_strain.copy(),
                "kinematic_fiber_strain": fiber_strain_all.copy(),
                "fiber_stress": stress.copy(),
                "axial_force": float(np.mean(axial_forces)) if axial_forces else 0.0,
                "equivalent_stress_measure": (
                    "hill48"
                    if _elastic_symmetry(material) == "orthotropic"
                    else "von_mises"
                ),
                **initial_state,
            }

        if not tangent:
            return T.T @ F_loc, None, trial_state
        return T.T @ F_loc, T.T @ K_loc @ T, trial_state

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return 18

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((3, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Quadratic beam element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        chord = coords[2] - coords[0]
        length = float(np.linalg.norm(chord))
        if length > _SMALL:
            midpoint_error = float(np.linalg.norm(coords[1] - 0.5 * (coords[0] + coords[2])))
            if midpoint_error > max(1.0e-8 * length, 1.0e-12):
                raise ValueError(
                    f"Quadratic beam element {self.element_id} is curved or has a non-midpoint middle node. "
                    "The current B3 formulation is straight-sided; use two-node beam segments for curved geometry."
                )
        return coords

    def compute_shape_functions(self, xi: float) -> Tuple[np.ndarray, np.ndarray]:
        N = np.array([xi * (xi - 1.0) / 2.0, 1.0 - xi**2, xi * (xi + 1.0) / 2.0], dtype=float)
        dN_dxi = np.array([xi - 0.5, -2.0 * xi, xi + 0.5], dtype=float)
        return N, dN_dxi

    def _beam_frame_and_transform(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        length = float(np.linalg.norm(coords[2] - coords[0]))
        if length < _SMALL:
            raise ValueError(f"Quadratic beam element {self.element_id} has near-zero length")
        e1 = (coords[2] - coords[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((18, 18), dtype=float)
        Rt = R.T
        for i in range(3):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((18, 18), dtype=float)
        if self.generalized_section is not None:
            section_stiffness = generalized_beam_stiffness(
                self.generalized_section
            )
            K = np.zeros((18, 18), dtype=float)
            for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
                shape, derivative_xi = self.compute_shape_functions(float(xi))
                B = _beam_generalized_strain_matrix_from_shape(
                    shape,
                    derivative_xi * 2.0 / L,
                )
                K += (
                    B.T
                    @ section_stiffness
                    @ B
                    * (L / 2.0 * float(weight))
                )
            K_global = T.T @ K @ T
            self._stiffness_matrix = K_global
            return K_global
        E, G12, G13, GJ = _beam_material_properties(material, self.cross_section)
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GA_y = G12 * self._A * self._ky
        GA_z = G13 * self._A * self._kz
        K = np.zeros((18, 18), dtype=float)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            N, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            B_axial = np.zeros((1, 18), dtype=float)
            B_torsion = np.zeros((1, 18), dtype=float)
            B_shear_xz = np.zeros((1, 18), dtype=float)
            B_shear_xy = np.zeros((1, 18), dtype=float)
            B_bend_y = np.zeros((1, 18), dtype=float)
            B_bend_z = np.zeros((1, 18), dtype=float)
            for i in range(3):
                b = i * 6
                B_axial[0, b + 0] = dN_dx[i]
                B_torsion[0, b + 3] = dN_dx[i]
                B_shear_xz[0, b + 2] = dN_dx[i]
                B_shear_xz[0, b + 4] = N[i]
                B_shear_xy[0, b + 1] = dN_dx[i]
                B_shear_xy[0, b + 5] = -N[i]
                B_bend_y[0, b + 4] = dN_dx[i]
                B_bend_z[0, b + 5] = dN_dx[i]
            jac = L / 2.0 * weight
            K += B_axial.T @ (EA * np.eye(1)) @ B_axial * jac
            K += B_torsion.T @ (GJ * np.eye(1)) @ B_torsion * jac
            K += B_shear_xz.T @ (GA_z * np.eye(1)) @ B_shear_xz * jac
            K += B_shear_xy.T @ (GA_y * np.eye(1)) @ B_shear_xy * jac
            K += B_bend_y.T @ (EIy * np.eye(1)) @ B_bend_y * jac
            K += B_bend_z.T @ (EIz * np.eye(1)) @ B_bend_z * jac
        K_global = T.T @ K @ T
        self._stiffness_matrix = K_global
        return K_global

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((18, 18), dtype=float)
        M = np.zeros((18, 18), dtype=float)
        generalized_mass = (
            None
            if self.generalized_section is None
            else generalized_beam_mass_matrix(self.generalized_section)
        )
        if generalized_mass is not None:
            for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
                N, _ = self.compute_shape_functions(float(xi))
                jacobian_weight = float(weight) * L / 2.0
                for i in range(3):
                    for j in range(3):
                        M[
                            i * 6 : (i + 1) * 6,
                            j * 6 : (j + 1) * 6,
                        ] += (
                            N[i]
                            * N[j]
                            * jacobian_weight
                            * generalized_mass
                        )
            M_global = T.T @ M @ T
            self._mass_matrix = M_global
            return M_global
        rho = material.density
        # Consistent rotary inertia per rotation axis: polar (Iy+Iz) for
        # torsion, the matching section inertia for each bending rotation.
        rotary_inertia = (rho * (self._Iy + self._Iz), rho * self._Iy, rho * self._Iz)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            N, _ = self.compute_shape_functions(float(xi))
            for i in range(3):
                for j in range(3):
                    translational = N[i] * N[j] * rho * self._A * weight * L / 2.0
                    for d in range(3):
                        M[i * 6 + d, j * 6 + d] += translational
                        M[i * 6 + 3 + d, j * 6 + 3 + d] += N[i] * N[j] * rotary_inertia[d] * weight * L / 2.0
        M_global = T.T @ M @ T
        self._mass_matrix = M_global
        return M_global

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Beam-column stress stiffness from the lateral displacement gradient:

            KG = N_compression * integral (dN/dx)^T (dN/dx) dx

        applied to both transverse deflections.  This is the same
        destabilizing-gradient theory as the shell membrane KG and follows the
        package convention ``K phi = lambda KG phi`` with compression positive.
        The higher-order rotation-gradient term is omitted, which makes the
        predicted critical loads slightly conservative on coarse meshes.
        """
        axial_compression = self._axial_compression_from_state(state)
        if axial_compression == 0.0:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        KG = np.zeros((18, 18), dtype=float)
        # Wagner term factor: see BeamElement.compute_geometric_stiffness_matrix.
        polar_ratio = self._geometric_polar_radius_squared()
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            _, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            G_matrix = np.zeros((2, 18), dtype=float)
            twist_gradient = np.zeros((1, 18), dtype=float)
            for i in range(3):
                b = i * 6
                G_matrix[0, b + 1] = dN_dx[i]
                G_matrix[1, b + 2] = dN_dx[i]
                twist_gradient[0, b + 3] = dN_dx[i]
            KG += axial_compression * (G_matrix.T @ G_matrix) * (L / 2.0 * weight)
            KG += axial_compression * polar_ratio * (twist_gradient.T @ twist_gradient) * (L / 2.0 * weight)
        return T.T @ KG @ T


class CoupledBeamShellElement(Element):
    """Kinematic MPC coupling between one eccentric beam node and one shell node."""

    def __init__(
        self,
        element_id: int,
        beam_node_id: int,
        shell_node_id: int,
        material_name: str = "default",
        coupling_stiffness: float = 0.0,
        eccentricity: Optional[np.ndarray] = None,
        *,
        rotating_eccentricity: bool = True,
    ):
        if type(rotating_eccentricity) is not bool:
            raise TypeError("rotating_eccentricity must be a bool")
        super().__init__(element_id, [beam_node_id, shell_node_id], material_name)
        self.rotating_eccentricity = rotating_eccentricity
        self.beam_node_id = beam_node_id
        self.shell_node_id = shell_node_id
        self.coupling_stiffness = coupling_stiffness
        self.eccentricity = None if eccentricity is None else np.asarray(eccentricity, dtype=float)
        if self.eccentricity is not None:
            self._validate_eccentricity_shape(self.eccentricity)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        beam_node = mesh.get_node(self.beam_node_id)
        shell_node = mesh.get_node(self.shell_node_id)
        if beam_node is None or shell_node is None:
            raise ValueError(f"Coupling element {self.element_id} references a missing node")
        return np.array([beam_node.coords(), shell_node.coords()], dtype=float)

    def _validate_eccentricity_shape(self, eccentricity: np.ndarray) -> np.ndarray:
        eccentricity = np.asarray(eccentricity, dtype=float)
        if eccentricity.shape != (3,):
            raise ValueError(
                f"Coupling element {self.element_id} eccentricity must have shape (3,), "
                f"got {eccentricity.shape}"
            )
        if not np.all(np.isfinite(eccentricity)):
            raise ValueError(
                f"Coupling element {self.element_id} eccentricity contains non-finite values"
            )
        return eccentricity

    def _eccentricity_vector(self, mesh: "FEMesh") -> np.ndarray:
        if self.eccentricity is not None:
            r = np.asarray(self.eccentricity, dtype=float)
        else:
            coords = self.get_node_coordinates(mesh)
            r = np.asarray(coords[0] - coords[1], dtype=float)
        return self._validate_eccentricity_shape(r)

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        # Coupling is enforced exactly by assembly.build_constraint_transformation().
        # Returning zero avoids penalty-stiffness pollution.
        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self._stiffness_matrix = K
        return K

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        """
        Return slave/master constraints for the eccentric beam-shell relation.

        Beam node DOFs are slaves. Shell node DOFs are masters:
            u_b = u_s + theta_s x r
            theta_b = theta_s
        where r points from shell node to beam node.
        """
        beam_node = mesh.get_node(self.beam_node_id)
        shell_node = mesh.get_node(self.shell_node_id)
        if beam_node is None:
            raise ValueError(
                f"Coupling element {self.element_id} references missing beam node "
                f"{self.beam_node_id}"
            )
        if shell_node is None:
            raise ValueError(
                f"Coupling element {self.element_id} references missing shell node "
                f"{self.shell_node_id}"
            )

        b = beam_node.dofs
        s = shell_node.dofs
        rx, ry, rz = self._eccentricity_vector(mesh)

        return [
            {
                "slave": b[0],
                "masters": {s[0]: 1.0, s[4]: rz, s[5]: -ry},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_x_{self.element_id}",
            },
            {
                "slave": b[1],
                "masters": {s[1]: 1.0, s[3]: -rz, s[5]: rx},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_y_{self.element_id}",
            },
            {
                "slave": b[2],
                "masters": {s[2]: 1.0, s[3]: ry, s[4]: -rx},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_z_{self.element_id}",
            },
            {"slave": b[3], "masters": {s[3]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_rx_{self.element_id}"},
            {"slave": b[4], "masters": {s[4]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_ry_{self.element_id}"},
            {"slave": b[5], "masters": {s[5]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_rz_{self.element_id}"},
        ]

    def get_nonlinear_mpc_descriptor(self, mesh: "FEMesh") -> Optional[Dict[str, Any]]:
        """Return the finite-rotation master-slave descriptor for corotational solves.

        Under finite shell rotation ``theta_s``, the eccentric offset ``r_0``
        rotates rigidly with the shell node:
            u_b     = u_s + (R(theta_s) - I) r_0
            theta_b = theta_s
        Constructor option ``rotating_eccentricity=False`` disables the finite-
        rotation lever arm and retains the fixed-reference linearized law.
        """
        rotating_eccentricity = getattr(self, "rotating_eccentricity", True)
        if type(rotating_eccentricity) is not bool:
            raise TypeError("rotating_eccentricity must be a bool")
        if not rotating_eccentricity:
            return None
        beam_node = mesh.get_node(self.beam_node_id)
        shell_node = mesh.get_node(self.shell_node_id)
        if beam_node is None:
            raise ValueError(
                f"Coupling element {self.element_id} references missing beam node "
                f"{self.beam_node_id}"
            )
        if shell_node is None:
            raise ValueError(
                f"Coupling element {self.element_id} references missing shell node "
                f"{self.shell_node_id}"
            )
        r = np.asarray(self._eccentricity_vector(mesh), dtype=float).copy()
        if float(np.linalg.norm(r)) == 0.0:
            return None
        b = beam_node.dofs
        s = shell_node.dofs
        return {
            "element_id": int(self.element_id),
            "slave_u_dofs": tuple(int(d) for d in b[:3]),
            "slave_theta_dofs": tuple(int(d) for d in b[3:6]),
            "masters": (
                (
                    tuple(int(d) for d in s[:3]),
                    tuple(int(d) for d in s[3:6]),
                    1.0,
                ),
            ),
            "eccentricity": r,
            "label_prefix": f"beam_shell_ecc_{self.element_id}",
        }

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)



def validate_initial_field_state(
    element: Element,
    material: "Material",
    state: Dict[str, Any],
    num_layers: int,
    mesh: Optional["FEMesh"] = None,
) -> None:
    """Validate that a prescribed initial stress is locally admissible.

    Residual-stress input specifies the stress *at the initialized reference
    state*; it is not a request for the solver to reconstruct an unknown
    manufacturing load path.  For nonlinear materials the supplied stress
    must therefore lie on or inside the flow surface associated with the
    supplied hardening state.
    """
    tolerance = 1.0e-9
    if isinstance(element, ShellElement) and (
        "initial_membrane_stress" in state or "initial_bending_stress" in state
    ):
        symmetry = _elastic_symmetry(material)
        curve = getattr(material, "hardening_curve", None)
        hill_yield = getattr(material, "hill_yield", None)
        if symmetry == "orthotropic" and curve is not None and hill_yield is None:
            raise ValueError(
                f"Orthotropic material {getattr(material, 'name', '<unnamed>')!r} "
                "requires hill_yield when hardening_curve is active."
            )
        if curve is None and hill_yield is None:
            return
        n_gp = len(element.gauss_points)
        z_layers, _weights = lobatto_layers(num_layers, element.thickness)
        zero = np.zeros((n_gp, 3), dtype=float)
        membrane = _shell_field_rows(
            state.get("initial_membrane_stress", zero),
            n_gp,
            "initial_membrane_stress",
        )
        bending = _shell_field_rows(
            state.get("initial_bending_stress", zero),
            n_gp,
            "initial_bending_stress",
        )
        stress = (
            membrane[:, None, :]
            + (2.0 * z_layers[None, :, None] / element.thickness) * bending[:, None, :]
        ).reshape(n_gp * num_layers, 3)
        alpha = np.asarray(state.get("alpha", np.zeros(stress.shape[0])), dtype=float).reshape(-1)
        if alpha.size != stress.shape[0]:
            raise ValueError(
                f"Shell initial alpha has {alpha.size} entries; expected {stress.shape[0]}"
            )
        if np.any(~np.isfinite(alpha)) or np.any(alpha < 0.0):
            raise ValueError("Shell initial alpha must be finite and non-negative")
        if symmetry == "orthotropic":
            from .plasticity import hill48_plane_stress_equivalent_stress

            if mesh is None:
                if element.material_direction is not None:
                    raise ValueError(
                        "mesh is required to validate an oriented orthotropic "
                        "shell initial stress"
                    )
                material_angle = np.deg2rad(element.material_angle_deg)
            else:
                coordinates = element.get_node_coordinates(mesh)
                material_angle = element._material_angle(
                    element._center_frame(coordinates)
                )
            _Q, _G, _strain_to_material, stress_to_local = (
                _shell_material_matrices(material, material_angle)
            )
            stress_material = np.einsum(
                "ij,nj->ni",
                np.linalg.inv(stress_to_local),
                stress,
            )
            equivalent = hill48_plane_stress_equivalent_stress(
                stress_material,
                hill_yield,
            )
            if curve is None:
                flow = np.full(equivalent.shape, float(hill_yield.X))
            else:
                reference = float(
                    np.asarray(curve.flow_stress(np.array([0.0])), dtype=float)[0]
                )
                flow = (
                    float(hill_yield.X)
                    * np.asarray(curve.flow_stress(alpha), dtype=float)
                    / reference
                )
        else:
            equivalent = np.sqrt(
                np.maximum(
                    stress[:, 0] ** 2
                    - stress[:, 0] * stress[:, 1]
                    + stress[:, 1] ** 2
                    + 3.0 * stress[:, 2] ** 2,
                    0.0,
                )
            )
            flow = np.asarray(curve.flow_stress(alpha), dtype=float)
        excess = equivalent - flow
        if np.any(excess > tolerance * np.maximum(flow, 1.0)):
            index = int(np.argmax(excess))
            raise ValueError(
                f"Initial shell stress on element {element.element_id} is outside the supplied "
                f"hardening-state flow surface at point {index}: "
                f"sigma_eq={equivalent[index]:.6g}, flow={flow[index]:.6g}. "
                "Supply an admissible stress or compatible alpha history."
            )
    if isinstance(element, (BeamElement, QuadraticBeamElement)) and "initial_fiber_stress" in state:
        config = element._fiber_plasticity_config(material)
        if config is None:
            return
        curve = config.material_curve
        y, _z, _weights = element._fiber_section_grid(config)
        section_fibers = y.size
        total_fibers = section_fibers * (
            len(element.GAUSS_POINTS) if isinstance(element, QuadraticBeamElement) else 1
        )
        stress = _beam_fiber_field(
            state["initial_fiber_stress"],
            total_fibers,
            section_fibers,
            "initial_fiber_stress",
        )
        alpha = np.asarray(state.get("alpha", np.zeros(total_fibers)), dtype=float).reshape(-1)
        if alpha.size not in {1, total_fibers}:
            raise ValueError(
                f"Beam initial alpha has {alpha.size} entries; expected 1 or {total_fibers}"
            )
        if alpha.size == 1:
            alpha = np.full(total_fibers, float(alpha[0]), dtype=float)
        if np.any(~np.isfinite(alpha)) or np.any(alpha < 0.0):
            raise ValueError("Beam initial alpha must be finite and non-negative")
        scale = _beam_flow_scale(material, curve)
        flow = (
            np.full(alpha.shape, scale, dtype=float)
            if curve is None
            else scale * np.asarray(curve.flow_stress(alpha), dtype=float)
        )
        excess = np.abs(stress) - flow
        if np.any(excess > tolerance * np.maximum(flow, 1.0)):
            index = int(np.argmax(excess))
            raise ValueError(
                f"Initial beam-fiber stress on element {element.element_id} is outside the supplied "
                f"hardening-state flow surface at fiber {index}: "
                f"|sigma|={abs(stress[index]):.6g}, flow={flow[index]:.6g}. "
                "Supply an admissible stress or compatible alpha history."
            )


ELEMENT_TYPES = {
    "shell": ShellElement,
    "shell3": ShellElement,
    "tri3": ShellElement,
    "tria3": ShellElement,
    "t3": ShellElement,
    "s3": ShellElement,
    "shell6": ShellElement,
    "tri6": ShellElement,
    "tria6": ShellElement,
    "t6": ShellElement,
    "s6": ShellElement,
    "beam": BeamElement,
    "quadratic_beam": QuadraticBeamElement,
    "coupled": CoupledBeamShellElement,
}


DEFAULT_Q4_FORMULATION = "e4-pl"
# New/current-policy three-node shells use the V2D companion accepted by the
# V6W qualification. Historical records without formulation identity still
# deserialize explicitly as ``legacy-s3`` below, and direct
# ``ShellElement(...)`` construction remains the documented legacy API.
DEFAULT_S3_FORMULATION = "e4-pl-s3-v2d"
LEGACY_Q4_AVAILABLE_THROUGH = "0.4.x"
LEGACY_Q4_REMOVAL_TARGET = "0.5.0"


class LegacyQ4DeprecationWarning(DeprecationWarning):
    """Warning emitted when the temporary legacy four-node rollback is used."""


class LegacyS3MigrationWarning(UserWarning):
    """Warning emitted when a missing-ID historical TRI3 loads as legacy."""


LegacyShellElement = ShellElement


def _normalized_shell_formulation(
    node_count: int,
    formulation: Optional[str],
) -> str:
    if formulation is None:
        resolved = (
            DEFAULT_Q4_FORMULATION
            if node_count == 4
            else DEFAULT_S3_FORMULATION
            if node_count == 3
            else "legacy"
        )
    else:
        resolved = str(formulation)
    raw_resolved = resolved.strip().lower()
    resolved = raw_resolved.replace("_", "-")
    exact_s3_v2_selectors = {"e4-pl-s3-v2", "qualified-s3-v2"}
    exact_s3_v2b_selectors = {"e4-pl-s3-v2b", "qualified-s3-v2b"}
    exact_s3_v2c_selectors = {"e4-pl-s3-v2c", "qualified-s3-v2c"}
    exact_s3_v2d_selectors = {"e4-pl-s3-v2d", "qualified-s3-v2d"}
    if resolved in exact_s3_v2_selectors and raw_resolved not in exact_s3_v2_selectors:
        raise ValueError(
            "strict-flat E4-PL S3 V2 requires the canonical e4-pl-s3-v2 selector"
        )
    if resolved in exact_s3_v2b_selectors and raw_resolved not in exact_s3_v2b_selectors:
        raise ValueError(
            "strict-flat E4-PL S3 V2B requires the canonical e4-pl-s3-v2b selector"
        )
    if resolved in exact_s3_v2c_selectors and raw_resolved not in exact_s3_v2c_selectors:
        raise ValueError(
            "strict-flat E4-PL S3 V2C requires the canonical e4-pl-s3-v2c selector"
        )
    if resolved in exact_s3_v2d_selectors and raw_resolved not in exact_s3_v2d_selectors:
        raise ValueError(
            "native-parity E4-PL S3 V2D requires the canonical e4-pl-s3-v2d selector"
        )
    if resolved == "default":
        resolved = (
            DEFAULT_Q4_FORMULATION
            if node_count == 4
            else DEFAULT_S3_FORMULATION
            if node_count == 3
            else "legacy"
        )
    e4_aliases = {"e4-pl", "qualified-s4"}
    s3_aliases = {"e4-pl-s3", "qualified-s3"}
    s3_v2_aliases = exact_s3_v2_selectors
    s3_v2b_aliases = exact_s3_v2b_selectors
    s3_v2c_aliases = exact_s3_v2c_selectors
    s3_v2d_aliases = exact_s3_v2d_selectors
    legacy_aliases = {"legacy", "legacy-shell", "legacy-s4"}
    if node_count == 4 and resolved in e4_aliases:
        return "e4-pl"
    if node_count == 3 and resolved in s3_aliases:
        return "e4-pl-s3-v2d"
    if node_count == 3 and resolved in s3_v2_aliases:
        return "e4-pl-s3-v2"
    if node_count == 3 and resolved in s3_v2b_aliases:
        return "e4-pl-s3-v2b"
    if node_count == 3 and resolved in s3_v2c_aliases:
        return "e4-pl-s3-v2c"
    if node_count == 3 and resolved in s3_v2d_aliases:
        return "e4-pl-s3-v2d"
    if resolved == "legacy-s3":
        if node_count != 3:
            raise ValueError("legacy-s3 is available only for three-node shell topology")
        return "legacy-s3"
    if resolved in legacy_aliases:
        return "legacy"
    if resolved in e4_aliases:
        raise ValueError("E4-PL is available only for four-node shell topology")
    if resolved in s3_aliases:
        raise ValueError("qualified E4-PL S3 is available only for three-node shell topology")
    if resolved in s3_v2_aliases:
        raise ValueError(
            "strict-flat E4-PL S3 V2 is available only for three-node shell topology"
        )
    if resolved in s3_v2b_aliases:
        raise ValueError(
            "strict-flat E4-PL S3 V2B is available only for three-node shell topology"
        )
    if resolved in s3_v2c_aliases:
        raise ValueError(
            "strict-flat E4-PL S3 V2C is available only for three-node shell topology"
        )
    if resolved in s3_v2d_aliases:
        raise ValueError(
            "native-parity E4-PL S3 V2D is available only for three-node shell topology"
        )
    raise ValueError(f"Unknown shell formulation: {formulation}")


def shell_formulation_diagnostics(
    *,
    node_count: int = 4,
    formulation: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe shell selection and the temporary legacy-Q4 rollback state.

    This helper performs no element construction or mechanics.  It is intended
    for migration tooling, logs, and support diagnostics.
    """

    count = int(node_count)
    if count not in {3, 4, 6, 8}:
        raise ValueError("shell node_count must be one of 3, 4, 6 or 8")
    selected = _normalized_shell_formulation(count, formulation)
    normalized_request = (
        None
        if formulation is None
        else str(formulation).strip().lower().replace("_", "-")
    )
    return {
        "schema": "anysolver.shell-formulation-diagnostics-v1",
        "node_count": count,
        "requested_formulation": normalized_request,
        "selected_formulation": selected,
        "production_default": formulation is None and count in {3, 4},
        "topology_policy": (
            "QUALIFIED_E4_PL_Q4"
            if count == 4 and selected == "e4-pl"
            else (
                "DEPRECATED_LEGACY_Q4_ROLLBACK"
                if count == 4
                else (
                    "QUALIFIED_E4_PL_S3_OPT_IN"
                    if count == 3 and selected == "e4-pl-s3"
                    else (
                        "STRICT_FLAT_LINEAR_E4_PL_S3_V2_OPT_IN"
                        if count == 3 and selected == "e4-pl-s3-v2"
                        else (
                            "STRICT_FLAT_LINEAR_E4_PL_S3_V2B_OPT_IN"
                            if count == 3 and selected == "e4-pl-s3-v2b"
                            else (
                                "STRICT_FLAT_LINEAR_E4_PL_S3_V2C_OPT_IN"
                                if count == 3 and selected == "e4-pl-s3-v2c"
                                else (
                                    "NATIVE_PARITY_E4_PL_S3_V2D_DEFAULT"
                                    if count == 3 and selected == "e4-pl-s3-v2d"
                                    else "PRESERVED_LEGACY_NON_Q4"
                                )
                            )
                        )
                    )
                )
            )
        ),
        "legacy_q4": {
            "state": "DEPRECATED_BURN_IN_ROLLBACK",
            "available_through": LEGACY_Q4_AVAILABLE_THROUGH,
            "removal_not_before": LEGACY_Q4_REMOVAL_TARGET,
            "selector": "legacy",
            "warning": "LegacyQ4DeprecationWarning",
        },
    }


def create_shell_element(
    element_id: int,
    node_ids: List[int],
    material_name: str = "default",
    *,
    formulation: Optional[str] = None,
    **kwargs: Any,
) -> ShellElement:
    """Create a topology-selected shell while retaining explicit rollbacks.

    Four-node and three-node shells select their qualified E4-PL formulations
    by default. Higher-order shell topologies retain
    :class:`LegacyShellElement`. Passing ``formulation="legacy-s3"`` is the
    documented three-node rollback; an explicit ``e4-pl`` request remains
    Q4-only.
    """

    resolved = _normalized_shell_formulation(len(node_ids), formulation)
    if resolved == "e4-pl":
        from .e4_pl_element import QualifiedE4PLShellElement

        return QualifiedE4PLShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved == "e4-pl-s3":
        from .e4_pl_s3_element import QualifiedE4PLS3ShellElement

        return QualifiedE4PLS3ShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved == "e4-pl-s3-v2":
        from .e4_pl_s3_v2_element import StrictFlatLinearE4PLS3V2ShellElement

        return StrictFlatLinearE4PLS3V2ShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved == "e4-pl-s3-v2b":
        from .e4_pl_s3_v2b_element import StrictFlatLinearE4PLS3V2BShellElement

        return StrictFlatLinearE4PLS3V2BShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved == "e4-pl-s3-v2c":
        from .e4_pl_s3_v2c_element import StrictFlatLinearE4PLS3V2CShellElement

        return StrictFlatLinearE4PLS3V2CShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved == "e4-pl-s3-v2d":
        from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement

        if kwargs.get("reference_normal") is None:
            raise ValueError(
                "qualified S3 V2D requires authoritative reference_normal; "
                "use formulation='legacy-s3' for historical connectivity-only input"
            )

        return NativeParityE4PLS3V2DShellElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if resolved in {"legacy", "legacy-s3"}:
        return LegacyShellElement(element_id, node_ids, material_name, **kwargs)
    raise AssertionError(f"unreachable shell formulation selection: {resolved}")


def shell_element_from_dict(payload: Mapping[str, Any]) -> ShellElement:
    """Deserialize a shell without guessing a qualified formulation.

    Historical three-node records without ``formulation_id`` remain legacy.
    Qualified records are dispatched by their exact identity and unknown
    identities fail closed.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("serialized shell element must be a mapping")
    data = dict(payload)
    node_ids = [int(value) for value in data.get("node_ids", ())]
    formulation_id = data.get("formulation_id")
    if formulation_id is not None:
        from .e4_pl_s3_v2_element import (
            FORMULATION_ID as S3_V2_FORMULATION_ID,
            StrictFlatLinearCapabilityError,
        )
        from .e4_pl_s3_v2b_element import (
            FORMULATION_ID as S3_V2B_FORMULATION_ID,
            StrictFlatLinearCapabilityError as StrictFlatLinearV2BCapabilityError,
        )
        from .e4_pl_s3_v2c_element import (
            FORMULATION_ID as S3_V2C_FORMULATION_ID,
            StrictFlatLinearE4PLS3V2CShellElement,
        )
        from .e4_pl_s3_v2d_element import (
            FORMULATION_ID as S3_V2D_FORMULATION_ID,
            NativeParityE4PLS3V2DShellElement,
        )

        if formulation_id == S3_V2_FORMULATION_ID:
            raise StrictFlatLinearCapabilityError(
                "strict-flat S3 V2 serialization/restart is outside the "
                "authorized Stage-1 surface"
            )
        if formulation_id == S3_V2B_FORMULATION_ID:
            raise StrictFlatLinearV2BCapabilityError(
                "strict-flat S3 V2B serialization/restart is outside the "
                "authorized production-parity surface"
            )
        if formulation_id == S3_V2C_FORMULATION_ID:
            return StrictFlatLinearE4PLS3V2CShellElement.from_dict(data)
        if formulation_id == S3_V2D_FORMULATION_ID:
            return NativeParityE4PLS3V2DShellElement.from_dict(data)
        from .e4_pl_element import (
            FORMULATION_ID as Q4_FORMULATION_ID,
            QualifiedE4PLShellElement,
            _PLANAR_FORMULATION_ID,
        )
        from .e4_pl_s3_element import (
            FORMULATION_ID as S3_FORMULATION_ID,
            QualifiedE4PLS3ShellElement,
        )

        if formulation_id in {Q4_FORMULATION_ID, _PLANAR_FORMULATION_ID}:
            return QualifiedE4PLShellElement.from_dict(data)
        if formulation_id == S3_FORMULATION_ID:
            return QualifiedE4PLS3ShellElement.from_dict(data)
        raise ValueError(f"unknown serialized shell formulation_id: {formulation_id}")
    def normalize_candidate_fingerprint(value: Any) -> str:
        return str(value).strip().lower().replace("_", "-")

    normalized_serialized_type = normalize_candidate_fingerprint(
        data.get("type", "")
    )
    if normalized_serialized_type in {
        "qualifiede4plshellelement",
        "qualifiede4pls3shellelement",
        "strictflatlineare4pls3v2shellelement",
        "strictflatlineare4pls3v2bshellelement",
        "strictflatlineare4pls3v2cshellelement",
        "nativeparitye4pls3v2dshellelement",
        "e4-pl",
        "e4-pl-s3",
        "e4-pl-s3-v2",
        "e4-pl-s3-v2b",
        "e4-pl-s3-v2c",
        "e4-pl-s3-v2d",
        "qualified-s3-v2",
        "qualified-s3-v2b",
        "qualified-s3-v2c",
        "qualified-s3-v2d",
        "qualified-s3",
    }:
        raise ValueError("serialized qualified shell is missing formulation_id")
    qualified_s3_markers = frozenset(
        {
            "algebraic_coordinate_policy",
            "bubble_convention",
            "dynamic_reduction_policy",
            "director_polarity",
            "director_polarity_policy_id",
            "director_reversal_transform_id",
            "formulation_schema",
            "geometric_stiffness_policy",
            "mass_moment_id",
            "quadrature_id",
            "quadrature_authority_id",
            "reference_normal",
            "reference_surface_offset",
            "reference_surface_offset_policy_id",
            "reference_surface_strain_transform_id",
            "reference_surface_mass_shift_id",
            "recovery_policy_id",
            "state_layout_id",
        }
    )
    strict_s3_v2_unique_markers = frozenset(
        {
            "director_policy_id",
            "equation_map_sha256",
            "pl_completion_policy_id",
            "primary_source_sha256",
            "resultant_policy_id",
            "section_policy_id",
            "source_contract_schema",
            "source_contract_sha256",
            "strict_flat_linear_scope_id",
        }
    )
    qualified_q4_markers = frozenset(
        {
            "activity_disposition_schema_id",
            "current_state_binding_schema_id",
            "current_state_algorithmic_origin_schema_id",
            "current_state_projection_policy_id",
            "current_state_tangent_decomposition_policy_id",
            "deleted_frozen_policy_id",
            "director_polarity",
            "director_polarity_policy_id",
            "director_reversal_transform_id",
            "implementation_id",
            "quadrature_authority_id",
            "failed_state_policy_id",
            "recovery_policy_id",
            "reference_normal",
            "stationary_solve_policy_id",
        }
    )
    strict_s3_v2_marker_set = set(strict_s3_v2_unique_markers & data.keys())
    serialized_selector = data.get("selector")
    if isinstance(serialized_selector, str) and normalize_candidate_fingerprint(
        serialized_selector
    ) in {
        "e4-pl-s3-v2",
        "e4-pl-s3-v2b",
        "e4-pl-s3-v2c",
        "e4-pl-s3-v2d",
        "qualified-s3-v2",
        "qualified-s3-v2b",
        "qualified-s3-v2c",
        "qualified-s3-v2d",
    }:
        strict_s3_v2_marker_set.add("selector")
    if normalize_candidate_fingerprint(data.get("implementation_id", "")) == (
        "e4-pl-s3-v2-dkmt-eq12-41-cst-pl-hammer3-v1"
    ):
        strict_s3_v2_marker_set.add("implementation_id")
    if normalize_candidate_fingerprint(data.get("implementation_id", "")) == (
        "e4-pl-s3-v2c-min3-relaxed-uhm-cst-pl-parity-v1"
    ):
        strict_s3_v2_marker_set.add("implementation_id")
    if normalize_candidate_fingerprint(data.get("implementation_id", "")) == (
        "e4-pl-s3-v2d-min3-native-section-linear-gate-v1"
    ):
        strict_s3_v2_marker_set.add("implementation_id")
    if normalize_candidate_fingerprint(data.get("implementation_id", "")) == (
        "e4-pl-s3-v2b-min3-relaxed-uhm-cst-pl-hammer3-v1"
    ):
        strict_s3_v2_marker_set.add("implementation_id")
    if normalize_candidate_fingerprint(data.get("formulation_schema", "")) == (
        "anysolver.e4-pl-s3-v2-strict-flat-linear-element-v1"
    ):
        strict_s3_v2_marker_set.add("formulation_schema")
    if normalize_candidate_fingerprint(data.get("formulation_schema", "")) == (
        "anysolver.e4-pl-s3-v2c-flat-linear-parity-element-v1"
    ):
        strict_s3_v2_marker_set.add("formulation_schema")
    if normalize_candidate_fingerprint(data.get("formulation_schema", "")) == (
        "anysolver.e4-pl-s3-v2d-native-parity-element-v1"
    ):
        strict_s3_v2_marker_set.add("formulation_schema")
    if normalize_candidate_fingerprint(data.get("formulation_schema", "")) == (
        "anysolver.e4-pl-s3-v2b-strict-flat-linear-element-v1"
    ):
        strict_s3_v2_marker_set.add("formulation_schema")
    if normalize_candidate_fingerprint(
        data.get("quadrature_authority_id", "")
    ) == "s3-v2-dkmt-hammer3-degree2-exact-v1":
        strict_s3_v2_marker_set.add("quadrature_authority_id")
    if strict_s3_v2_marker_set:
        raise ValueError(
            "serialized shell retains strict-flat S3 V2 fingerprint markers "
            "but is missing formulation_id: "
            + ", ".join(sorted(strict_s3_v2_marker_set))
        )
    retained_s3_marker_set = set(qualified_s3_markers & data.keys())
    retained_s3_markers = tuple(sorted(retained_s3_marker_set))
    if len(node_ids) == 3 and retained_s3_markers:
        raise ValueError(
            "serialized three-node shell retains qualified S3 fingerprint "
            "markers but is missing formulation_id: "
            + ", ".join(retained_s3_markers)
        )
    retained_q4_markers = tuple(sorted(qualified_q4_markers & data.keys()))
    if len(node_ids) == 4 and retained_q4_markers:
        raise ValueError(
            "serialized four-node shell retains qualified Q4 fingerprint "
            "markers but is missing formulation_id: "
            + ", ".join(retained_q4_markers)
        )
    if len(node_ids) not in {3, 4, 6, 8}:
        raise ValueError("serialized legacy shell requires 3, 4, 6 or 8 nodes")
    element = create_shell_element(
        element_id=int(data["element_id"]),
        node_ids=node_ids,
        material_name=str(data.get("material_name", "default")),
        formulation="legacy-s3" if len(node_ids) == 3 else "legacy",
        thickness=float(data.get("thickness", 0.01)),
        drilling_stabilization=float(data.get("drilling_stabilization", 1.0e-3)),
        reduced_integration=bool(data.get("reduced_integration", False)),
        hourglass_stabilization=float(data.get("hourglass_stabilization", 1.0e-8)),
        material_direction=data.get("material_direction"),
        material_angle_deg=float(data.get("material_angle_deg", 0.0)),
        shell_section=data.get("shell_section"),
    )
    if len(node_ids) == 3:
        warnings.warn(
            "LEGACY_S3_MISSING_FORMULATION_ID: historical three-node shell "
            "record loaded as legacy-s3; qualified-s3 was not inferred; "
            "qualified-s3 hot restart is forbidden without full load-history "
            "replay",
            LegacyS3MigrationWarning,
            stacklevel=2,
        )
    return element


def create_element(
    element_type: str,
    element_id: int,
    node_ids: List[int],
    material_name: str = "default",
    **kwargs: Any,
) -> Element:
    normalized_type = str(element_type).lower()
    if normalized_type == "ge-beam3":
        # The qualified geometrically exact three-node beam is deliberately
        # kept out of ``ELEMENT_TYPES``.  That registry also feeds legacy
        # beam dispatch and batching paths, while GE-B3 has its own rotation
        # transaction and state contract.  Keep this one exact opt-in route
        # local so no existing beam spelling changes meaning.
        from .ge_beam3_element import GeometricallyExactBeam3D3NElement

        return GeometricallyExactBeam3D3NElement(
            element_id,
            node_ids,
            material_name,
            **kwargs,
        )
    if normalized_type in {"e4-pl-s3-v2d", "qualified-s3-v2d"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl-s3-v2d",
            **kwargs,
        )
    if normalized_type in {"e4-pl-s3-v2c", "qualified-s3-v2c"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl-s3-v2c",
            **kwargs,
        )
    if normalized_type in {"e4-pl-s3-v2b", "qualified-s3-v2b"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl-s3-v2b",
            **kwargs,
        )
    if normalized_type in {"e4-pl-s3-v2", "qualified-s3-v2"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl-s3-v2",
            **kwargs,
        )
    if normalized_type in {"e4-pl-s3", "e4_pl_s3", "qualified-s3", "qualified_s3"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl-s3",
            **kwargs,
        )
    if normalized_type in {"legacy-s3", "legacy_s3"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="legacy-s3",
            **kwargs,
        )
    if normalized_type in {"e4-pl", "e4_pl", "qualified_s4"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="e4-pl",
            **kwargs,
        )
    if normalized_type in {"legacy-shell", "legacy_shell", "legacy-s4", "legacy_s4"}:
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation="legacy",
            **kwargs,
        )
    if normalized_type == "shell":
        formulation = kwargs.pop("formulation", None)
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation=formulation,
            **kwargs,
        )
    if normalized_type in {"shell3", "tri3", "tria3", "t3", "s3"}:
        formulation = kwargs.pop("formulation", None)
        return create_shell_element(
            element_id,
            node_ids,
            material_name,
            formulation=formulation,
            **kwargs,
        )
    if normalized_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type: {element_type}")
    if normalized_type == "coupled":
        if len(node_ids) != 2:
            raise ValueError("CoupledBeamShellElement factory requires [beam_node_id, shell_node_id]")
        return CoupledBeamShellElement(element_id, node_ids[0], node_ids[1], material_name, **kwargs)
    return ELEMENT_TYPES[normalized_type](element_id, node_ids, material_name, **kwargs)
