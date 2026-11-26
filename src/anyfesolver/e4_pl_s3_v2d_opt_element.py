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

import inspect
import math
import weakref
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .e4_pl_opt_membrane import (
    RECOVERY_E0,
    isotropic_plane_stress,
    optimal_beta0,
    opt_membrane_stiffness,
    opt_membrane_strain_operator,
)
from .e4_pl_s3_v2c_element import (
    HAMMER_POINTS,
    HAMMER_REFERENCE_WEIGHTS,
    RESULTANT_POLICY_ID,
    StrictFlatLinearE4PLS3V2CShellElement as _V2C,
)
from .e4_pl_s3_v2d_element import (
    NativeParityCapabilityError,
    CBMIN3,
    NativeParityE4PLS3V2DShellElement as _V2D,
    _ROTATIONAL_INDICES,
    _immutable_serialized_unit_vector,
    _immutable_vector,
    _real_scalar,
)
from .elements import ShellElement
from .fe_core import FEMesh, Material, _QualifiedMutationEpoch, _QualifiedStateMapping

SELECTOR = "e4-pl-s3-v2d-opt"
FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_OPT_V1"
FORMULATION_SCHEMA = "ANYFESOLVER_E4_PL_S3_V2D_OPT_ELEMENT_V1"
IMPLEMENTATION_ID = "E4_PL_S3_V2D_OPT_LINEAR_V1"
MEMBRANE_POLICY_ID = "FELIPPA_2003_ANDES_OPT_LST39R_V1"
SCOPE_POLICY_ID = "LINEAR_STATIC_AND_RECOVERY_ONLY_V1"
_LOCAL_CACHE: Dict[Any, Dict[str, Any]] = {}
_LOCAL_CACHE_LIMIT = 1024
_MATERIAL_CACHE: Dict[Any, Any] = {}
_LOCAL_MEMBRANE_INDEX = np.asarray((0, 1, 5, 6, 7, 11, 12, 13, 17), dtype=np.intp)  # (u,v,rz) per node
_OPT_FIELDS = (
    "element_id", "formulation_id", "formulation_schema", "implementation_id", "material_name", "node_ids",
    "reference_normal", "selector", "membrane_policy_id", "scope_policy_id", "thickness", "type",
)


_SCOPE_CACHE: list = []  # [(weakref(elements), token, epoch, length)] of the last verified registry


def _require_opt_only_shell_scope(elements: Any) -> None:
    """Refuse a model holding any shell other than V2D-OPT.

    The class census is linear in mesh size; per owner contract 2026-11-24 the
    registry does not change mid-request in supported use, so a verified census
    is reused while the registry's mutation epoch and length are unchanged.
    """
    key = None
    if type(elements) is _QualifiedStateMapping:
        token = dict.get(object.__getattribute__(elements, "__dict__"), "_qualified_token")
        if type(token) is _QualifiedMutationEpoch:
            key = (token, list.__getitem__(token, 0), dict.__len__(elements))
            if _SCOPE_CACHE:
                ref, token0, epoch0, length0 = _SCOPE_CACHE[0]
                if ref() is elements and token0 is key[0] and epoch0 == key[1] and length0 == key[2]:
                    return
    kinds = {}
    for element in dict.values(elements):
        kind = type(element)
        kinds[id(kind)] = kind  # identity census: never hashes/compares the classes
    for kind in kinds.values():
        if kind is not NativeParityE4PLS3V2DOptShellElement and issubclass(kind, ShellElement):
            raise NativeParityCapabilityError("S3 V2D-OPT mixed shell scope admits only V2D-OPT shells")
    if key is not None:
        try:
            _SCOPE_CACHE[:] = [(weakref.ref(elements), *key)]
        except TypeError:
            _SCOPE_CACHE.clear()


def _plain_float(value: Any, label: str) -> float:
    """Exact Python float of a finite real: never evaluates caller-supplied arithmetic."""
    return float(np.float64(_real_scalar(value, label)))


def _snapshot_constitutive(material: Material, thickness: float) -> Dict[str, Any]:
    """Elastic isotropic section from plain-float snapshots (same expressions as the V2D baseline)."""
    modulus = _plain_float(getattr(material, "elastic_modulus", None), "elastic_modulus")
    poisson = _plain_float(getattr(material, "poisson_ratio", None), "poisson_ratio")
    if modulus <= 0.0 or not -1.0 < poisson < 0.5:
        raise ValueError("S3 V2D-OPT isotropic elastic constants are inadmissible")
    scale = modulus / (1.0 - poisson**2)
    plane = scale * np.asarray(
        ((1.0, poisson, 0.0), (poisson, 1.0, 0.0), (0.0, 0.0, 0.5 * (1.0 - poisson))), dtype=np.float64
    )
    shear_scalar = (5.0 / 6.0) * modulus * thickness / (2.0 * (1.0 + poisson))
    return {
        "A": thickness * plane, "B": np.zeros((3, 3), dtype=np.float64), "D": thickness**3 / 12.0 * plane,
        "H": shear_scalar * np.eye(2, dtype=np.float64), "E": modulus, "nu": poisson,
        "drill_scale": thickness * modulus / (2.0 * (1.0 + poisson)),
    }


_HAMMER_REFERENCE = (HAMMER_POINTS.copy(), HAMMER_POINTS.strides, HAMMER_REFERENCE_WEIGHTS.copy(), HAMMER_REFERENCE_WEIGHTS.strides)


def _require_intact_quadrature() -> None:
    points, point_strides, weights, weight_strides = _HAMMER_REFERENCE
    if (HAMMER_POINTS.strides != point_strides or HAMMER_REFERENCE_WEIGHTS.strides != weight_strides
            or not np.array_equal(HAMMER_POINTS, points) or not np.array_equal(HAMMER_REFERENCE_WEIGHTS, weights)):
        raise NativeParityCapabilityError("S3 V2D-OPT shared quadrature arrays were altered")


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

    # Reviewed V2D/V2C functions used unchanged (unbound functions bind to this class).
    _membrane_operator = staticmethod(_V2C._membrane_operator)
    _pl_constraint = staticmethod(_V2C._pl_constraint)
    _coordinates = _V2D._coordinates
    _elastic_baseline_constitutive = _V2D._elastic_baseline_constitutive
    _is_v2d_elastic_overlap = staticmethod(_V2D._is_v2c_elastic_overlap)
    _globalize = staticmethod(_V2D._globalize)
    _pressure_virtual_work_density_local = staticmethod(_V2C._pressure_virtual_work_density_local)
    compute_native_surface_shape_functions = staticmethod(_V2D.compute_native_surface_shape_functions)
    sheet_area_orientation_sign = _V2D.sheet_area_orientation_sign
    compute_dead_transverse_pressure_load = _V2D.compute_dead_transverse_pressure_load
    compute_native_dead_pressure_load = _V2D.compute_native_dead_pressure_load
    compute_native_pressure_density = _V2D.compute_native_pressure_density
    physical_reference_director = _V2D.physical_reference_director
    native_reference_directors = _V2D.native_reference_directors
    material_mid_surface_offset_from_reference = _V2D.material_mid_surface_offset_from_reference
    material_surface_offset_from_reference = _V2D.material_surface_offset_from_reference

    @property
    def TRI_GAUSS_POINTS_3(self) -> np.ndarray:
        return HAMMER_POINTS.copy()

    @property
    def TRI_GAUSS_WEIGHTS_3(self) -> np.ndarray:
        return HAMMER_REFERENCE_WEIGHTS.copy()

    @property
    def gauss_points(self) -> np.ndarray:
        return HAMMER_POINTS.copy()

    @property
    def gauss_weights(self) -> np.ndarray:
        return HAMMER_REFERENCE_WEIGHTS.copy()

    @property
    def shear_gauss_points(self) -> np.ndarray:
        return HAMMER_POINTS.copy()

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        return HAMMER_REFERENCE_WEIGHTS.copy()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _PROTECTED_NAMES:
            raise NativeParityCapabilityError(f"S3 V2D-OPT forbids assigning the protected name {name!r}")
        super().__setattr__(name, value)

    def _geometry(self, mesh: FEMesh) -> Dict[str, Any]:
        _require_no_shadow(self)
        return _V2D._geometry(self, mesh)

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
        _require_opt_only_shell_scope(elements)

    def _require_material(self, material: Material) -> None:
        if not self._is_v2d_elastic_overlap(material):
            raise NativeParityCapabilityError("S3 V2D-OPT admits exact elastic isotropic Material only")

    # ---------------------------------------------------------------- local data (cached)
    def _local_data(self, geometry: Mapping[str, Any], material: Material) -> Dict[str, Any]:
        """Element-local (internal node order) blocks, operators and MIN3 relaxation.

        Pure function of (local coordinates, thickness, E, nu); results are cached by those exact
        float64 values (bit-identical reuse; congruent triangles share an entry).  Bending and
        shear blocks use the reviewed V2C operators with the operation order of the V2D native
        path; the membrane block is the OPT kernel.
        """
        _require_no_shadow(self)
        thickness = _plain_float(self.thickness, "thickness")
        raw = _snapshot_constitutive(material, thickness)
        local = np.asarray(geometry["local_coordinates"], dtype=np.float64)
        key = (local.tobytes(), thickness, raw["E"], raw["nu"])
        hit = _LOCAL_CACHE.get(key)
        if hit is not None:
            return hit
        _require_intact_quadrature()  # never fill the cache while a shared rule array is altered
        ops = _V2C._operators(self, geometry, raw)
        curvature, shear_op = ops["B_b"], ops["B_s"]
        weight = float(geometry["area"]) / 3.0
        bending = np.zeros((18, 18), dtype=np.float64)
        unrelaxed = np.zeros((18, 18), dtype=np.float64)
        for b_op, s_op in zip(curvature, shear_op):
            bending += b_op.T @ raw["D"] @ b_op * weight
            unrelaxed += s_op.T @ raw["H"] @ s_op * weight
        bending_sum = float(sum(bending[i, i] for i in _ROTATIONAL_INDICES))
        shear_sum = float(sum(unrelaxed[i, i] for i in _ROTATIONAL_INDICES))
        if bending_sum <= 0.0 or shear_sum <= 0.0:
            raise ValueError("S3 V2D-OPT MIN3 relaxation traces must be positive")
        psi_hat = bending_sum / shear_sum
        phi_squared = CBMIN3 * psi_hat / (1.0 + CBMIN3 * psi_hat)
        if not math.isfinite(phi_squared) or not 0.0 < phi_squared <= 1.0:
            raise ValueError("S3 V2D-OPT MIN3 relaxation is invalid")
        plane_key = (float(raw["E"]), float(raw["nu"]))
        material_entry = _MATERIAL_CACHE.get(plane_key)
        if material_entry is None:
            plane = isotropic_plane_stress(*plane_key)
            material_entry = (plane, float(optimal_beta0(plane)))
            if len(_MATERIAL_CACHE) >= _LOCAL_CACHE_LIMIT:
                _MATERIAL_CACHE.clear()
            _MATERIAL_CACHE[plane_key] = material_entry
        plane, beta0 = material_entry
        membrane_local = np.zeros((18, 18), dtype=np.float64)
        membrane_local[np.ix_(_LOCAL_MEMBRANE_INDEX, _LOCAL_MEMBRANE_INDEX)] = opt_membrane_stiffness(
            local, plane, thickness, beta0, validated=True
        )
        bary = np.column_stack(
            (1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1], HAMMER_POINTS[:, 0], HAMMER_POINTS[:, 1])
        )
        membrane_ops = np.zeros((3, 3, 18), dtype=np.float64)
        membrane_ops[:, :, _LOCAL_MEMBRANE_INDEX] = opt_membrane_strain_operator(
            local, bary, thickness, RECOVERY_E0
        )
        entry = {
            "membrane": membrane_local, "bending": bending, "shear": phi_squared * unrelaxed,
            "phi_squared": phi_squared, "phi": np.asarray(ops["phi"], dtype=np.float64),
            "B_m": membrane_ops, "B_b": np.asarray(curvature, dtype=np.float64),
            "B_s": np.asarray(shear_op, dtype=np.float64), "section": raw,
            "beta0": beta0,
        }
        for value in (*entry.values(), *raw.values()):
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        if not all(np.all(np.isfinite(entry[name])) for name in ("membrane", "bending", "shear", "B_m", "B_b", "B_s")):
            raise ValueError("S3 V2D-OPT local stiffness is not finite")
        if len(_LOCAL_CACHE) >= _LOCAL_CACHE_LIMIT:
            _LOCAL_CACHE.clear()
        _LOCAL_CACHE[key] = entry
        return entry

    # ---------------------------------------------------------------- stiffness
    def compute_stiffness_components(self, mesh: FEMesh, material: Material) -> Mapping[str, Any]:
        self._validate_configuration()
        self._require_material(material)
        geometry = self._geometry(mesh)
        data = self._local_data(geometry, material)
        transform = np.asarray(geometry["local_from_external"], dtype=np.float64)
        zero = np.zeros((18, 18), dtype=np.float64)
        physical = self._globalize(data["membrane"] + data["bending"] + data["shear"], transform)
        if not np.all(np.isfinite(physical)):
            raise ValueError("S3 V2D-OPT global stiffness is not finite")
        return {
            "membrane": self._globalize(data["membrane"], transform),
            "membrane_bending_coupling": zero.copy(),
            "bending": self._globalize(data["bending"], transform),
            "shear": self._globalize(data["shear"], transform),
            "physical": physical, "pl": zero.copy(), "numerical": zero.copy(), "hourglass": zero.copy(),
            "total": physical.copy(), "frame": np.asarray(geometry["frame"], dtype=np.float64).copy(),
            "area": float(geometry["area"]), "phi": data["phi"].copy(), "phi_squared": data["phi_squared"],
            "drill_scale": 0.0, "formulation_id": FORMULATION_ID, "membrane_policy_id": MEMBRANE_POLICY_ID,
            "membrane_beta0": data["beta0"],
        }

    def compute_stiffness_matrix(self, mesh: FEMesh, material: Material) -> np.ndarray:
        self._validate_configuration()
        self._require_material(material)
        geometry = self._geometry(mesh)
        data = self._local_data(geometry, material)
        total = data["membrane"] + data["bending"] + data["shear"]
        made = self._globalize(total, np.asarray(geometry["local_from_external"], dtype=np.float64))
        if not np.all(np.isfinite(made)):
            raise ValueError("S3 V2D-OPT global stiffness is not finite")
        return made

    def compute_internal_forces(self, mesh: FEMesh, displacements: np.ndarray, material: Material) -> np.ndarray:
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError("S3 V2D-OPT displacement must resolve to a finite 18-vector")
        forces = self.compute_stiffness_matrix(mesh, material) @ vector
        if not np.all(np.isfinite(forces)):
            raise ValueError("S3 V2D-OPT internal forces are not finite")
        return forces

    # ---------------------------------------------------------------- recovery
    def compute_variational_resultants(
        self, mesh: FEMesh, displacements: np.ndarray, material: Material
    ) -> Dict[str, Any]:
        self._validate_configuration()
        self._require_material(material)
        geometry = self._geometry(mesh)
        data = self._local_data(geometry, material)
        section = data["section"]
        vector = self._get_element_displacements(mesh, displacements)
        if vector.shape != (18,) or not np.all(np.isfinite(vector)):
            raise ValueError("S3 V2D-OPT displacement must resolve to a finite 18-vector")
        local_vector = np.asarray(geometry["local_from_external"], dtype=np.float64) @ vector
        epsilon = np.einsum("gij,j->gi", data["B_m"], local_vector)
        kappa = np.einsum("gij,j->gi", data["B_b"], local_vector)
        gamma = np.einsum("gij,j->gi", data["B_s"], local_vector)
        phi_squared = float(data["phi_squared"])
        if not (np.all(np.isfinite(epsilon)) and np.all(np.isfinite(kappa)) and np.all(np.isfinite(gamma))):
            raise ValueError("S3 V2D-OPT recovered strains are not finite")
        membrane_resultants = epsilon @ section["A"].T
        bending_resultants = kappa @ section["D"].T
        shear_resultants = phi_squared * gamma @ section["H"].T
        if not (np.all(np.isfinite(membrane_resultants)) and np.all(np.isfinite(bending_resultants))
                and np.all(np.isfinite(shear_resultants))):
            raise ValueError("S3 V2D-OPT recovered resultants are not finite")
        order = tuple(int(value) for value in geometry["internal_order"])

        def external_order(values: np.ndarray) -> np.ndarray:
            made = np.empty_like(values)
            for internal, external in enumerate(order):
                made[external] = values[internal]
            return made

        barycentric = np.column_stack(
            (1.0 - HAMMER_POINTS[:, 0] - HAMMER_POINTS[:, 1], HAMMER_POINTS[:, 0], HAMMER_POINTS[:, 1])
        )
        return {
            "recovery_scope": RESULTANT_POLICY_ID, "qualified_recovery": False,
            "physical_stress_available": False, "numerical_fields_excluded": True,
            "membrane_strain": external_order(epsilon), "curvature": external_order(kappa),
            "transverse_shear_strain": external_order(gamma),
            "membrane_resultants": external_order(membrane_resultants),
            "bending_resultants": external_order(bending_resultants),
            "transverse_shear_resultants": external_order(shear_resultants),
            "min3_relaxation_phi_squared": phi_squared,
            "hammer_points": np.asarray(HAMMER_POINTS, dtype=np.float64).copy(),
            "external_barycentric_coordinates": barycentric,
            "physical_station_coordinates": barycentric
            @ np.asarray(geometry["external_coordinates"], dtype=np.float64),
            "internal_order": order,
            "physical_weights": np.full(3, float(geometry["area"]) / 3.0),
            "frame": np.asarray(geometry["frame"], dtype=np.float64).copy(),
            "resultant_policy_id": RESULTANT_POLICY_ID,
            "native_section_policy_id": "S3_V2D_LAYERED_ELASTIC_BASELINE_V1",
            "director_polarity": 1, "reference_surface_offset": 0.0,
            "section_origin_offset_from_reference": 0.0,
            "physical_bottom_offset_from_reference": self.material_surface_offset_from_reference(-1.0),
            "physical_top_offset_from_reference": self.material_surface_offset_from_reference(1.0),
        }

    def compute_stresses(
        self, mesh: FEMesh, displacements: np.ndarray, material: Material, return_global: bool = False
    ):
        self._require_material(material)
        result = dict(_V2D.compute_stresses(self, mesh, displacements, material, return_global))
        for name, value in result.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError(f"S3 V2D-OPT recovered field {name!r} is not finite")
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
    def __getstate__(self) -> Dict[str, Any]:
        _refuse("python pickle/copy (use to_dict/from_dict)")

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _refuse("python pickle/copy (use to_dict/from_dict)")

    def __reduce_ex__(self, protocol: int) -> Any:
        _refuse("python pickle/copy (use to_dict/from_dict)")

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        _refuse("python deepcopy (use to_dict/from_dict)")

    def __copy__(self) -> Any:
        _refuse("python copy (use to_dict/from_dict)")

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
            "thickness": _plain_float(self.thickness, "thickness"),
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


def _protected_names() -> frozenset:
    names = {"__class__", "__dict__", "director_polarity", "reference_surface_offset", "formulation_id", "selector", "implementation_id",
             "_legacy_shell_dispatch_forbidden", "node_ids", "reference_normal"}
    cls = NativeParityE4PLS3V2DOptShellElement
    for name in dir(cls):
        static = inspect.getattr_static(cls, name)
        if callable(static) or isinstance(static, (staticmethod, classmethod, property)):
            names.add(name)
    # data fields set by the constructor stay assignable by the constructor only
    names.discard("node_ids")
    names.discard("reference_normal")
    return frozenset(names)


_PROTECTED_NAMES = _protected_names()


def _require_no_shadow(element: Any) -> None:
    """Refuse an instance whose own namespace shadows any class method/policy constant."""
    namespace = object.__getattribute__(element, "__dict__")
    if not _PROTECTED_NAMES.isdisjoint(namespace):
        raise NativeParityCapabilityError("S3 V2D-OPT instance attribute shadows a route or policy name")
