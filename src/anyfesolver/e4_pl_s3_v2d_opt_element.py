"""Opt-in S3 triangle: V2D plate action with the Felippa OPT membrane (linear route only).

Formulation id ``e4-pl-s3-v2d-opt``.  Bending, transverse shear, frame, geometry
and pressure loads are the reviewed V2D/V2C functions themselves (bound, not
re-authored); the membrane block ``CST + PL drill penalty`` is replaced by the
Felippa 2003 OPT membrane with its own drilling stiffness
(``e4_pl_opt_membrane``).  This class is deliberately NOT a V2D subclass: the
default V2D admission guards are exact-type and stay untouched.

Admitted: linear static stiffness, dead pressure loads, stress/resultant
recovery, serialization.  Everything else (nonlinear/current-state, restart
state, contact, mass/geometric stiffness, reference offset, reversed polarity,
generalized sections, plastic or non-isotropic materials, mixed shell scope)
is refused with ``NativeParityCapabilityError``.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .e4_pl_opt_membrane import (
    RECOVERY_E0,
    isotropic_plane_stress,
    optimal_beta0,
    opt_membrane_stiffness,
    opt_membrane_strain_operator,
)
from .e4_pl_s3_v2c_element import HAMMER_POINTS, StrictFlatLinearE4PLS3V2CShellElement as _V2C
from .e4_pl_s3_v2d_element import (
    NativeParityCapabilityError,
    NativeParityE4PLS3V2DShellElement as _V2D,
    _immutable_serialized_unit_vector,
    _immutable_vector,
    _real_scalar,
)
from .elements import ShellElement
from .fe_core import FEMesh, Material

SELECTOR = "e4-pl-s3-v2d-opt"
FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_OPT_V1"
FORMULATION_SCHEMA = "ANYFESOLVER_E4_PL_S3_V2D_OPT_ELEMENT_V1"
IMPLEMENTATION_ID = "E4_PL_S3_V2D_OPT_LINEAR_V1"
MEMBRANE_POLICY_ID = "FELIPPA_2003_ANDES_OPT_LST39R_V1"
SCOPE_POLICY_ID = "LINEAR_STATIC_AND_RECOVERY_ONLY_V1"
_LOCAL_MEMBRANE_INDEX = np.asarray((0, 1, 5, 6, 7, 11, 12, 13, 17), dtype=np.intp)  # (u,v,rz) per node
_OPT_FIELDS = (
    "element_id", "formulation_id", "formulation_schema", "implementation_id", "material_name", "node_ids",
    "reference_normal", "selector", "membrane_policy_id", "scope_policy_id", "thickness", "type",
)


def _refuse(operation: str) -> None:
    raise NativeParityCapabilityError(f"S3 V2D-OPT linear v1 does not admit {operation}")


class NativeParityE4PLS3V2DOptShellElement(ShellElement):
    """Flat 3-node shell: V2D MIN3 plate action + Felippa OPT membrane with drilling."""

    formulation_id = FORMULATION_ID
    implementation_id = IMPLEMENTATION_ID
    selector = SELECTOR
    legacy_stiffness_batch_eligible = False
    legacy_nonlinear_batch_eligible = False
    formulation_native_total_lagrangian = False
    _legacy_shell_dispatch_forbidden = FORMULATION_ID
    director_polarity = 1
    reference_surface_offset = 0.0
    TRI_GAUSS_POINTS_3 = _V2D.TRI_GAUSS_POINTS_3
    TRI_GAUSS_WEIGHTS_3 = _V2D.TRI_GAUSS_WEIGHTS_3

    # Reviewed V2D/V2C functions used unchanged (unbound functions bind to this class).
    _membrane_operator = staticmethod(_V2C._membrane_operator)
    _pl_constraint = staticmethod(_V2C._pl_constraint)
    _geometry = _V2D._geometry
    _coordinates = _V2D._coordinates
    _elastic_baseline_constitutive = _V2D._elastic_baseline_constitutive
    _is_v2d_elastic_overlap = staticmethod(_V2D._is_v2c_elastic_overlap)
    # The reused V2D recovery/stiffness functions delegate to V2C when this predicate holds;
    # OPT must always take the native path (its membrane operator differs).
    _is_v2c_elastic_overlap = staticmethod(lambda material: False)
    _director_adjusted_section = _V2D._director_adjusted_section
    _effective_station_operators = _V2D._effective_station_operators
    _reference_surface_strain_transform = _V2D._reference_surface_strain_transform
    _globalize = staticmethod(_V2D._globalize)
    _native_generalized_components = _V2D._native_generalized_components
    _pressure_virtual_work_density_local = staticmethod(_V2C._pressure_virtual_work_density_local)
    compute_native_surface_shape_functions = staticmethod(_V2D.compute_native_surface_shape_functions)
    sheet_area_orientation_sign = _V2D.sheet_area_orientation_sign
    compute_dead_transverse_pressure_load = _V2D.compute_dead_transverse_pressure_load
    compute_native_dead_pressure_load = _V2D.compute_native_dead_pressure_load
    compute_native_pressure_density = _V2D.compute_native_pressure_density
    compute_variational_resultants = _V2D.compute_variational_resultants
    physical_reference_director = _V2D.physical_reference_director
    native_reference_directors = _V2D.native_reference_directors
    material_mid_surface_offset_from_reference = _V2D.material_mid_surface_offset_from_reference
    material_surface_offset_from_reference = _V2D.material_surface_offset_from_reference

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        material_name: str = "default",
        *,
        thickness: float = 0.01,
        reference_normal: Sequence[float],
        director_polarity: int = 1,
        reference_surface_offset: float = 0.0,
        material_direction: Optional[Sequence[float]] = None,
        material_angle_deg: float = 0.0,
        shell_section: Optional[Any] = None,
    ) -> None:
        if type(element_id) is not int:
            raise TypeError("S3 V2D-OPT element_id must be an exact integer")
        owned = tuple(node_ids)
        if len(owned) != 3 or not all(type(n) is int for n in owned) or len(set(owned)) != 3:
            raise ValueError("S3 V2D-OPT requires three distinct exact-integer nodes")
        if type(director_polarity) is not int or director_polarity != 1:
            _refuse("director_polarity other than +1")
        if _real_scalar(reference_surface_offset, "reference_surface_offset") != 0.0:
            _refuse("a nonzero reference-surface offset")
        if shell_section is not None:
            _refuse("generalized shell sections")
        if material_direction is not None or float(material_angle_deg) != 0.0:
            _refuse("material orientation (isotropic only)")
        h = _real_scalar(thickness, "thickness")
        if h <= 0.0:
            raise ValueError("S3 V2D-OPT thickness must be positive")
        normal = _immutable_vector(reference_normal, "reference_normal")
        super().__init__(
            element_id, owned, material_name, thickness=h, drilling_stabilization=0.0,
            reduced_integration=False, hourglass_stabilization=0.0,
            material_direction=None, material_angle_deg=0.0, shell_section=None,
        )
        self.node_ids = owned
        self.reference_normal = normal

    # ---------------------------------------------------------------- admission
    def _validate_configuration(self) -> None:
        if type(self) is not NativeParityE4PLS3V2DOptShellElement:
            raise NativeParityCapabilityError("S3 V2D-OPT requires its exact class")
        if self.formulation_id != FORMULATION_ID or self.selector != SELECTOR:
            raise NativeParityCapabilityError("S3 V2D-OPT class identity changed")
        if type(self.node_ids) is not tuple or len(self.node_ids) != 3 or len(set(self.node_ids)) != 3:
            raise ValueError("S3 V2D-OPT connectivity authority changed")
        if self.drilling_stabilization != 0.0 or self.hourglass_stabilization != 0.0 or self.reduced_integration is not False:
            raise NativeParityCapabilityError("S3 V2D-OPT forbids legacy stabilizations")
        if self.shell_section is not None or self.material_direction is not None or float(self.material_angle_deg) != 0.0:
            raise NativeParityCapabilityError("S3 V2D-OPT is isotropic and sectionless")
        if self.director_polarity != 1 or self.reference_surface_offset != 0.0:
            raise NativeParityCapabilityError("S3 V2D-OPT polarity/offset authority changed")
        normal = self.reference_normal
        if (type(normal) is not np.ndarray or normal.shape != (3,) or normal.dtype != np.dtype(np.float64)
                or normal.flags.writeable or not np.all(np.isfinite(normal))
                or abs(float(np.linalg.norm(normal)) - 1.0) > 8.0 * np.finfo(np.float64).eps):
            raise ValueError("S3 V2D-OPT reference-normal authority changed")

    def _validate_model_scope(self, mesh: FEMesh) -> None:
        if type(mesh) is not FEMesh:
            raise TypeError("S3 V2D-OPT requires an exact FEMesh")
        elements = object.__getattribute__(mesh, "__dict__").get("elements")
        if not elements:
            return
        if not isinstance(elements, dict) or elements.get(self.element_id) is not self:
            raise NativeParityCapabilityError("S3 V2D-OPT evaluation requires exact model-registry ownership")
        for other in elements.values():
            if isinstance(other, ShellElement) and type(other) is not NativeParityE4PLS3V2DOptShellElement:
                raise NativeParityCapabilityError("S3 V2D-OPT mixed shell scope admits only V2D-OPT shells")

    def _require_material(self, material: Material) -> None:
        if not self._is_v2d_elastic_overlap(material):
            raise NativeParityCapabilityError("S3 V2D-OPT admits exact elastic isotropic Material only")

    # ---------------------------------------------------------------- operators
    def _operators(self, geometry: Mapping[str, Any], constitutive: Mapping[str, Any]) -> Dict[str, Any]:
        """V2C plate operators; B_m is the per-station OPT strain-recovery operator (3,3,18)."""
        ops = _V2C._operators(self, geometry, constitutive)
        local = np.asarray(geometry["local_coordinates"], dtype=np.float64)
        bary = np.column_stack((1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1], HAMMER_POINTS[:, 0], HAMMER_POINTS[:, 1]))
        recovery = opt_membrane_strain_operator(local, bary, float(self.thickness), RECOVERY_E0)
        membrane = np.zeros((3, 3, 18), dtype=np.float64)
        membrane[:, :, _LOCAL_MEMBRANE_INDEX] = recovery
        ops = dict(ops)
        ops["B_m"] = membrane
        return ops

    # ---------------------------------------------------------------- stiffness
    def compute_stiffness_components(self, mesh: FEMesh, material: Material) -> Mapping[str, Any]:
        self._validate_configuration()
        self._require_material(material)
        parts = dict(self._native_generalized_components(mesh, material))
        geometry = self._geometry(mesh)
        raw = self._elastic_baseline_constitutive(material)
        plane = isotropic_plane_stress(raw["E"], raw["nu"])
        k9 = opt_membrane_stiffness(np.asarray(geometry["local_coordinates"], dtype=np.float64), plane, float(self.thickness))
        membrane_local = np.zeros((18, 18), dtype=np.float64)
        membrane_local[np.ix_(_LOCAL_MEMBRANE_INDEX, _LOCAL_MEMBRANE_INDEX)] = k9
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        membrane = self._globalize(membrane_local, transform)
        zero = np.zeros((18, 18), dtype=np.float64)
        physical = membrane + parts["membrane_bending_coupling"] + parts["bending"] + parts["shear"]
        parts.update(
            membrane=membrane, pl=zero.copy(), numerical=zero.copy(), physical=physical,
            total=0.5 * (physical + physical.T), drill_scale=0.0,
            formulation_id=FORMULATION_ID, membrane_policy_id=MEMBRANE_POLICY_ID,
            membrane_beta0=float(optimal_beta0(plane)),
        )
        return parts

    def compute_stiffness_matrix(self, mesh: FEMesh, material: Material) -> np.ndarray:
        return np.asarray(self.compute_stiffness_components(mesh, material)["total"], dtype=np.float64).copy()

    def compute_internal_forces(self, mesh: FEMesh, displacements: np.ndarray, material: Material) -> np.ndarray:
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError("S3 V2D-OPT displacement must resolve to a finite 18-vector")
        return self.compute_stiffness_matrix(mesh, material) @ vector

    # ---------------------------------------------------------------- recovery
    def compute_stresses(self, mesh: FEMesh, displacements: np.ndarray, material: Material, return_global: bool = False):
        self._require_material(material)
        result = dict(_V2D.compute_stresses(self, mesh, displacements, material, return_global))
        result["formulation_id"] = FORMULATION_ID
        result["implementation_id"] = IMPLEMENTATION_ID
        result["membrane_policy_id"] = MEMBRANE_POLICY_ID
        return result

    # ---------------------------------------------------------------- refused routes
    def compute_mass_matrix(self, *args: Any, **kwargs: Any):
        _refuse("mass matrices")

    def compute_geometric_stiffness_matrix(self, *args: Any, **kwargs: Any):
        _refuse("geometric stiffness")

    def compute_nonlinear_response(self, *args: Any, **kwargs: Any):
        _refuse("nonlinear response")

    def init_nonlinear_state(self, *args: Any, **kwargs: Any):
        _refuse("nonlinear state")

    def compute_native_current_pressure_load(self, *args: Any, **kwargs: Any):
        _refuse("current-configuration pressure")

    def compute_native_current_pressure_tangent(self, *args: Any, **kwargs: Any):
        _refuse("current-configuration pressure tangent")

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        self._validate_configuration()
        return {
            "element_id": int(self.element_id),
            "formulation_id": FORMULATION_ID,
            "formulation_schema": FORMULATION_SCHEMA,
            "implementation_id": IMPLEMENTATION_ID,
            "material_name": self.material_name,
            "node_ids": [int(v) for v in self.node_ids],
            "reference_normal": np.asarray(self.reference_normal).tolist(),
            "selector": SELECTOR,
            "membrane_policy_id": MEMBRANE_POLICY_ID,
            "scope_policy_id": SCOPE_POLICY_ID,
            "thickness": float(self.thickness),
            "type": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeParityE4PLS3V2DOptShellElement":
        if cls is not NativeParityE4PLS3V2DOptShellElement:
            raise NativeParityCapabilityError("S3 V2D-OPT deserialization requires exact class")
        if not isinstance(payload, Mapping):
            raise TypeError("S3 V2D-OPT serialized payload must be a mapping")
        data = dict(payload)
        if set(data) != set(_OPT_FIELDS):
            raise ValueError("S3 V2D-OPT serialized schema keys mismatch")
        identities = {
            "formulation_id": FORMULATION_ID, "formulation_schema": FORMULATION_SCHEMA,
            "implementation_id": IMPLEMENTATION_ID, "selector": SELECTOR,
            "membrane_policy_id": MEMBRANE_POLICY_ID, "scope_policy_id": SCOPE_POLICY_ID,
            "type": "NativeParityE4PLS3V2DOptShellElement",
        }
        if any(data[k] != v for k, v in identities.items()):
            raise ValueError("S3 V2D-OPT serialized fingerprint mismatch")
        normal = _immutable_serialized_unit_vector(data["reference_normal"], "reference_normal")
        restored = cls(data["element_id"], data["node_ids"], data["material_name"],
                       thickness=data["thickness"], reference_normal=normal)
        restored.reference_normal = normal
        restored._validate_configuration()
        return restored
