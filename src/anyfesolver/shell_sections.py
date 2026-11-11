"""Pre-integrated generalized shell-section constitutive data.

The section uses engineering in-plane order ``[11, 22, 12]`` and transverse
shear order ``[13, 23]``.  Its constitutive law is

.. code-block:: text

    N = A epsilon + B kappa
    M = B.T epsilon + D kappa
    Q = As gamma

``A`` and ``As`` therefore have units N/m, ``B`` has units N, and ``D`` has
units N*m.  ``As`` is the complete section shear stiffness and already
contains any shear-correction convention chosen by the section producer.

Only generalized resultants can be recovered from these pre-integrated
matrices.  Ply stresses require a layered section model and are deliberately
outside this contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MemberDescriptorType
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

SHELL_MEMBRANE_VOIGT_ORDER: Tuple[str, ...] = ("11", "22", "12")
SHELL_TRANSVERSE_SHEAR_ORDER: Tuple[str, ...] = ("13", "23")


@runtime_checkable
class GeneralizedShellSectionProtocol(Protocol):
    """Structural contract accepted by :class:`~anyfesolver.ShellElement`.

    External section repositories can satisfy this protocol without importing
    or subclassing ANYfeSolver.  Optional metadata such as ``name`` and mass
    properties is read structurally by :func:`coerce_generalized_shell_section`.
    """

    @property
    def A(self) -> Any:
        """Membrane stiffness in N/m, shape ``(3, 3)``."""

    @property
    def B(self) -> Any:
        """Membrane-bending coupling stiffness in N, shape ``(3, 3)``."""

    @property
    def D(self) -> Any:
        """Bending stiffness in N*m, shape ``(3, 3)``."""

    @property
    def As(self) -> Any:
        """Complete transverse-shear stiffness in N/m, shape ``(2, 2)``."""


def _finite_matrix(value: Any, shape: Tuple[int, int], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Generalized shell section {label} must be a numeric {shape} matrix") from exc
    if matrix.shape != shape:
        raise ValueError(
            f"Generalized shell section {label} must have shape {shape}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Generalized shell section {label} must contain only finite values")
    return np.array(matrix, dtype=float, copy=True)


def _immutable_float64_matrix(matrix: np.ndarray) -> np.ndarray:
    """Own one matrix through an immutable bytes buffer.

    ``setflags(write=False)`` is reversible for an owning NumPy array.  The
    generalized-section contract is immutable, so retain its validated values
    in a bytes-backed array whose write flag cannot subsequently be enabled.
    """

    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(contiguous.shape)


def _require_symmetric(matrix: np.ndarray, label: str) -> np.ndarray:
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12 * scale):
        raise ValueError(f"Generalized shell section {label} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _require_positive_definite(matrix: np.ndarray, label: str) -> None:
    """Check positive definiteness after a diagonal congruence scaling.

    ABD matrices mix N/m, N and N*m entries.  Scaling to unit diagonal avoids
    treating that expected unit disparity as numerical ill-conditioning while
    preserving definiteness.
    """

    diagonal = np.diag(matrix)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError(f"Generalized shell section {label} must be positive definite")
    roots = np.sqrt(diagonal)
    scaled = matrix / np.outer(roots, roots)
    scaled = 0.5 * (scaled + scaled.T)
    try:
        np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Generalized shell section {label} must be positive definite"
        ) from exc


def _optional_nonnegative(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Generalized shell section {label} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"Generalized shell section {label} must be finite and non-negative")
    return result


def _in_plane_transforms(angle_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return engineering strain, resultant and shear-vector transforms."""

    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("Generalized shell section rotation angle must be finite")
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    section_axes_in_local = np.array([[c, -s], [s, c]], dtype=float)

    strain_to_section = np.zeros((3, 3), dtype=float)
    resultant_to_local = np.zeros((3, 3), dtype=float)
    for column in range(3):
        engineering_strain = np.zeros(3, dtype=float)
        engineering_strain[column] = 1.0
        local_strain_tensor = np.array(
            [
                [engineering_strain[0], 0.5 * engineering_strain[2]],
                [0.5 * engineering_strain[2], engineering_strain[1]],
            ],
            dtype=float,
        )
        section_strain_tensor = (
            section_axes_in_local.T @ local_strain_tensor @ section_axes_in_local
        )
        strain_to_section[:, column] = (
            section_strain_tensor[0, 0],
            section_strain_tensor[1, 1],
            2.0 * section_strain_tensor[0, 1],
        )

        section_resultant = np.zeros(3, dtype=float)
        section_resultant[column] = 1.0
        section_resultant_tensor = np.array(
            [
                [section_resultant[0], section_resultant[2]],
                [section_resultant[2], section_resultant[1]],
            ],
            dtype=float,
        )
        local_resultant_tensor = (
            section_axes_in_local
            @ section_resultant_tensor
            @ section_axes_in_local.T
        )
        resultant_to_local[:, column] = (
            local_resultant_tensor[0, 0],
            local_resultant_tensor[1, 1],
            local_resultant_tensor[0, 1],
        )
    return strain_to_section, resultant_to_local, section_axes_in_local


@dataclass(frozen=True)
class GeneralizedShellSection:
    """Validated pre-integrated linear shell section.

    Parameters are expressed in the section 1/2 axes.  A shell element's
    ``material_direction`` and ``material_angle_deg`` orient those axes within
    the element surface.

    ``mass_per_area`` has units kg/m².  ``rotary_inertia_per_area`` is the
    through-thickness mass second moment per unit area, ``integral(rho*z² dz)``,
    with units kg.  A value of ``None`` independently selects the legacy
    homogeneous values ``rho*h`` and ``rho*h³/12`` from the attached material
    and element thickness.
    """

    A: Any
    B: Any
    D: Any
    As: Any
    name: str = ""
    mass_per_area: Optional[float] = None
    rotary_inertia_per_area: Optional[float] = None

    def __deepcopy__(self, memo: dict[int, Any]) -> "GeneralizedShellSection":
        """Return a namespace-neutral copy with independently frozen arrays.

        Avoiding the generic ``copyreg`` reconstruction path is important:
        that path may install ``__slotnames__`` on this authority-bound class.
        The four constitutive arrays remain immutable bytes-backed values in
        the copy rather than becoming writable NumPy owners.
        """

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
                if name in {"A", "B", "D", "As"} and type(value) is np.ndarray:
                    value_id = id(value)
                    if value_id in memo:
                        copied = memo[value_id]
                    else:
                        copied = _immutable_float64_matrix(value)
                        memo[value_id] = copied
                else:
                    copied = copy.deepcopy(value, memo)
                target_namespace[name] = copied

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

    def __post_init__(self) -> None:
        A = _require_symmetric(_finite_matrix(self.A, (3, 3), "A"), "A")
        B = _finite_matrix(self.B, (3, 3), "B")
        D = _require_symmetric(_finite_matrix(self.D, (3, 3), "D"), "D")
        As = _require_symmetric(_finite_matrix(self.As, (2, 2), "As"), "As")

        abd = np.block([[A, B], [B.T, D]])
        _require_positive_definite(abd, "ABD matrix")
        _require_positive_definite(As, "As matrix")

        A, B, D, As = (
            _immutable_float64_matrix(matrix) for matrix in (A, B, D, As)
        )
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "As", As)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "mass_per_area",
            _optional_nonnegative(self.mass_per_area, "mass_per_area"),
        )
        object.__setattr__(
            self,
            "rotary_inertia_per_area",
            _optional_nonnegative(
                self.rotary_inertia_per_area,
                "rotary_inertia_per_area",
            ),
        )

    @property
    def ABD(self) -> np.ndarray:
        """Return a new symmetric 6x6 generalized membrane/bending matrix."""

        return np.block([[self.A, self.B], [self.B.T, self.D]])

    def rotated(self, angle_rad: float) -> "GeneralizedShellSection":
        """Return the section stiffness rotated into shell-local axes."""

        strain_to_section, resultant_to_local, shear_axes = _in_plane_transforms(
            angle_rad
        )
        generalized_strain = np.block(
            [
                [strain_to_section, np.zeros((3, 3), dtype=float)],
                [np.zeros((3, 3), dtype=float), strain_to_section],
            ]
        )
        generalized_resultant = np.block(
            [
                [resultant_to_local, np.zeros((3, 3), dtype=float)],
                [np.zeros((3, 3), dtype=float), resultant_to_local],
            ]
        )
        abd_local = generalized_resultant @ self.ABD @ generalized_strain
        abd_local = 0.5 * (abd_local + abd_local.T)
        As_local = shear_axes @ self.As @ shear_axes.T
        return GeneralizedShellSection(
            A=abd_local[:3, :3],
            B=abd_local[:3, 3:],
            D=abd_local[3:, 3:],
            As=0.5 * (As_local + As_local.T),
            name=self.name,
            mass_per_area=self.mass_per_area,
            rotary_inertia_per_area=self.rotary_inertia_per_area,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible section definition."""

        return {
            "name": self.name,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "D": self.D.tolist(),
            "As": self.As.tolist(),
            "mass_per_area": self.mass_per_area,
            "rotary_inertia_per_area": self.rotary_inertia_per_area,
        }


def coerce_generalized_shell_section(
    section: Any,
) -> Optional[GeneralizedShellSection]:
    """Validate and copy a mapping or structurally compatible section object."""

    if section is None:
        return None
    if isinstance(section, GeneralizedShellSection):
        return section

    if isinstance(section, Mapping):
        source = section

        def read(label: str, default: Any = None) -> Any:
            return source[label] if label in source else default

    else:

        def read(label: str, default: Any = None) -> Any:
            return getattr(section, label, default)

    missing = [label for label in ("A", "B", "D", "As") if read(label) is None]
    if missing:
        raise TypeError(
            "Generalized shell section must provide A, B, D and As; missing "
            + ", ".join(missing)
        )
    return GeneralizedShellSection(
        A=read("A"),
        B=read("B"),
        D=read("D"),
        As=read("As"),
        name=read("name", ""),
        mass_per_area=read("mass_per_area"),
        rotary_inertia_per_area=read("rotary_inertia_per_area"),
    )


def validate_generalized_shell_section(section: Any) -> GeneralizedShellSection:
    """Return a validated solver-owned copy of ``section``.

    Unlike :func:`coerce_generalized_shell_section`, ``None`` is rejected.
    """

    validated = coerce_generalized_shell_section(section)
    if validated is None:
        raise TypeError("Generalized shell section cannot be None")
    return validated


__all__ = [
    "GeneralizedShellSection",
    "GeneralizedShellSectionProtocol",
    "SHELL_MEMBRANE_VOIGT_ORDER",
    "SHELL_TRANSVERSE_SHEAR_ORDER",
    "coerce_generalized_shell_section",
    "validate_generalized_shell_section",
]
