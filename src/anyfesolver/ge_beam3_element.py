"""Qualified opt-in facade for the straight mixed GE-Beam3.

The mechanics live in :mod:`anyfesolver.ge_beam3_mixed_element` and are the
accepted P2 implementation.  This module changes no equation or numerical
path.  It supplies the P3 public integration boundary: the qualified
formulation identity, standard linear routes, qualified committed state, and
strict element persistence.

Only the exact ``ge-beam3`` selector may construct this class through the
public factory.  Legacy B2/B3 routing and batching remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ._native_rotation_state import NativeElementRotationView
from .beam_sections import (
    GeneralizedBeamSection,
    generalized_beam_mass_matrix,
    generalized_beam_stiffness,
)
from .ge_beam3_mixed_element import (
    GE_BEAM3_MIXED_CANDIDATE_ID,
    GE_BEAM3_MIXED_CONDENSATION_ID,
    GE_BEAM3_MIXED_QUADRATURE_ID,
    GE_BEAM3_MIXED_REFERENCE_ID,
    GE_BEAM3_MIXED_ROTATION_ID,
    GeBeam3MixedCommittedStateError,
    GeBeam3MixedGeometryError,
    GeBeam3MixedStateError,
    GeometricallyExactBeam3D3NElement as _CandidateGeBeam3,
)
from .ge_beam3_state import (
    GE_BEAM3_QUALIFIED_FORMULATION_ID,
    SECTION_SCHEMA,
    STATE_LAYOUT_ID,
    GeBeam3CommittedStateError,
    canonical_json_bytes,
    decode_typed_array,
    encode_typed_array,
    initialize_ge_beam3_state,
    normalize_ge_beam3_checkpoint_state,
    serialize_ge_beam3_state,
    strict_canonical_json_loads,
    validate_ge_beam3_state,
)


GE_BEAM3_SELECTOR = "ge-beam3"
GE_BEAM3_FORMULATION_SCHEMA = "GE_BEAM3_MIXED_K1_MACRO_SCHEMA_V2"
GE_BEAM3_QUALIFICATION_ORIGIN = {
    "candidate_id": GE_BEAM3_MIXED_CANDIDATE_ID,
    "p2_closeout_commit": "e31c9e292a2fc9f6b57472bb8c5b90919a535492",
    "p2_terminal": "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY",
}

_ELEMENT_KEYS = frozenset(
    {
        "condensation_id",
        "element_id",
        "formulation_id",
        "formulation_schema",
        "material_name",
        "node_ids",
        "qualification_origin",
        "quadrature_id",
        "reference_axis_direction",
        "reference_geometry",
        "reference_id",
        "reference_orientation",
        "reference_triad",
        "rotation_id",
        "section_descriptor",
        "state_layout_id",
        "type",
    }
)
_SECTION_KEYS = frozenset(
    {
        "mass_matrix_per_reference_length",
        "name",
        "schema",
        "stiffness_matrix",
    }
)


class GeBeam3IntegrationError(GeBeam3MixedStateError):
    """The requested route is outside the qualified P3 integration boundary."""


def _exact_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise GeBeam3IntegrationError(f"{label} must be an integer, not a boolean")
    return value


def _exact_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise GeBeam3IntegrationError(f"{label} must be a string")
    return value


def _readonly_array(
    value: Any,
    *,
    dtype: str,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise GeBeam3IntegrationError(f"{label} is not an array") from exc
    if source.shape != shape:
        raise GeBeam3IntegrationError(f"{label} must have shape {shape}")
    if dtype == "<i8":
        if source.dtype.kind not in "iu" or source.dtype.kind == "b":
            raise GeBeam3IntegrationError(f"{label} must contain integers")
    elif source.dtype.kind not in "fiu" or source.dtype.kind == "b":
        raise GeBeam3IntegrationError(f"{label} must contain real values")
    try:
        made = np.array(source, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeBeam3IntegrationError(f"{label} cannot be represented") from exc
    if made.dtype.kind == "f" and not np.all(np.isfinite(made)):
        raise GeBeam3IntegrationError(f"{label} contains nonfinite values")
    if made.dtype.kind == "i" and not np.array_equal(made, source):
        raise GeBeam3IntegrationError(f"{label} is outside signed int64")
    made.setflags(write=False)
    return made


def _same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


class GeometricallyExactBeam3D3NElement(_CandidateGeBeam3):
    """Qualified straight two-cell mixed Simo--Reissner beam.

    The public element has 18 external coordinates and uses the already
    accepted mixed local condensation.  Standard linear assembly sees only
    the condensed reference operator.  Finite rotations are evaluated solely
    through a solver-owned :class:`NativeElementRotationView`.
    """

    candidate_id = GE_BEAM3_MIXED_CANDIDATE_ID
    qualification_origin_candidate_id = GE_BEAM3_MIXED_CANDIDATE_ID
    formulation_id = GE_BEAM3_QUALIFIED_FORMULATION_ID
    formulation_schema = GE_BEAM3_FORMULATION_SCHEMA
    state_layout_id = STATE_LAYOUT_ID
    selector = GE_BEAM3_SELECTOR
    solver_integration_authorized = True
    # The generic native-store flag names the S3 director-triad extension,
    # not generic multiplicative-rotation consistency.  GE-B3 keeps it false:
    # the store still reconstructs and cross-checks committed_total_u plus the
    # node-shared rotation matrices, while this facade validates its complete
    # v2 state (including recovered station fields) before mechanics.
    native_state_consistency_required = False
    user_facing_name = "GE-B3 straight — mixed Simo–Reissner"

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        material_name: str = "default",
        cross_section: Optional[Mapping[str, Any]] = None,
        section: Any = None,
        reference_orientation: Any = None,
        reference_axis_direction: Any = None,
    ) -> None:
        super().__init__(
            element_id,
            node_ids,
            material_name,
            cross_section=cross_section,
            section=section,
            reference_orientation=reference_orientation,
            reference_axis_direction=reference_axis_direction,
        )
        # A qualified P3 record binds the complete stiffness-and-mass section
        # descriptor.  Requiring it here keeps modal capability and restart
        # identity from depending on a later optional input.
        if generalized_beam_mass_matrix(self.generalized_section) is None:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 requires an explicit symmetric positive-definite "
                "generalized 6x6 mass matrix per unit reference length"
            )
        self._qualified_reference_geometry: Optional[np.ndarray] = None
        self._qualified_reference_triad: Optional[np.ndarray] = None

    def _bind_reference_snapshot(
        self,
        geometry: Any,
        triad: Any,
        *,
        serialized: bool,
    ) -> None:
        made_geometry = _readonly_array(
            geometry,
            dtype="<f8",
            shape=(3, 3),
            label="reference_geometry",
        )
        made_triad = _readonly_array(
            triad,
            dtype="<f8",
            shape=(3, 3),
            label="reference_triad",
        )
        chord = made_geometry[2] - made_geometry[0]
        length = float(np.linalg.norm(chord))
        if not np.isfinite(length) or length <= 1.0e-14:
            raise GeBeam3MixedGeometryError(
                "qualified GE-B3 reference line has zero or nonfinite length"
            )
        if np.linalg.norm(made_geometry[1] - 0.5 * (made_geometry[0] + made_geometry[2])) > 1.0e-12 * length:
            raise GeBeam3MixedGeometryError(
                "qualified GE-B3 requires an exact straight-reference midpoint"
            )
        e1 = chord / length
        orientation = np.asarray(self.reference_orientation, dtype=np.float64)
        projected = orientation - float(orientation @ e1) * e1
        if np.linalg.norm(projected) <= 1.0e-12 * np.linalg.norm(orientation):
            raise GeBeam3MixedGeometryError(
                "qualified GE-B3 reference orientation is parallel to the beam axis"
            )
        e2 = projected / np.linalg.norm(projected)
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)
        expected = np.column_stack((e1, e2, e3)).astype("<f8", copy=False)
        scale = max(1.0, float(np.linalg.norm(expected, ord=np.inf)))
        if not np.allclose(made_triad, expected, rtol=0.0, atol=1.0e-12 * scale):
            raise GeBeam3MixedGeometryError(
                "qualified GE-B3 persisted triad disagrees with geometry and orientation"
            )
        if self.reference_axis_direction is not None:
            axis = np.asarray(self.reference_axis_direction, dtype=np.float64)
            alignment = float(axis @ e1) / float(np.linalg.norm(axis))
            if abs(abs(alignment) - 1.0) > 1.0e-12:
                raise GeBeam3MixedGeometryError(
                    "qualified GE-B3 reference axis is not parallel to its line"
                )

        if self._qualified_reference_geometry is None:
            self._qualified_reference_geometry = made_geometry
            self._qualified_reference_triad = made_triad
            return
        assert self._qualified_reference_triad is not None
        if not _same_bits(self._qualified_reference_geometry, made_geometry):
            disposition = "serialized " if serialized else "model "
            raise GeBeam3MixedGeometryError(
                f"qualified GE-B3 {disposition}reference geometry identity mismatch"
            )
        if not _same_bits(self._qualified_reference_triad, made_triad):
            disposition = "serialized " if serialized else "model "
            raise GeBeam3MixedGeometryError(
                f"qualified GE-B3 {disposition}reference triad identity mismatch"
            )

    def _reference_geometry(
        self, mesh: Any
    ) -> tuple[np.ndarray, float, np.ndarray, int]:
        # The existing coupling element equates additive nodal rotations.  It
        # is not the separately qualified objective shared-rotation/MPC joint
        # required by GE-B3.  Reject a connected instance before evaluating
        # this element's formulation-native operator.
        from .elements import CoupledBeamShellElement
        from .mesh_gen import InterpolatedBeamShellMPCElement

        own_nodes = frozenset(int(value) for value in self.node_ids)
        for other_id, other in getattr(mesh, "elements", {}).items():
            if other is self or not isinstance(
                other,
                (CoupledBeamShellElement, InterpolatedBeamShellMPCElement),
            ):
                continue
            if own_nodes.intersection(int(value) for value in other.node_ids):
                raise GeBeam3IntegrationError(
                    "qualified GE-B3 beam-shell connection is not authorized; "
                    f"coupling element {int(other_id)} requires a separately "
                    "qualified objective shared-rotation/MPC contract"
                )
        try:
            result = super()._reference_geometry(mesh)
        except GeBeam3MixedGeometryError as exc:
            if "two equal straight reference cells" in str(exc):
                raise GeBeam3MixedGeometryError(
                    "qualified GE-B3 straight-reference midpoint is not the "
                    "exact midpoint of its end nodes"
                ) from exc
            raise
        self._bind_reference_snapshot(result[0], result[2], serialized=False)
        return result

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        """Return the accepted P2 condensed reference-linear operator."""

        del material
        return np.array(self.compute_candidate_stiffness_matrix(mesh), copy=True)

    def compute_internal_forces(
        self,
        mesh: Any,
        displacements: Any,
        material: Any,
    ) -> np.ndarray:
        """Return exactly ``K_reference @ u`` for one finite 18-vector."""

        values = np.asarray(displacements, dtype=np.float64)
        if values.shape != (18,) or not np.all(np.isfinite(values)):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 linear displacement must contain 18 finite values"
            )
        matrix = self.compute_stiffness_matrix(mesh, material)
        return np.asarray(matrix @ values, dtype=np.float64)

    def compute_geometric_stiffness_matrix(
        self,
        mesh: Any,
        material: Any,
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """Map the exact reference-elastic Euler state into the P2 operator.

        Global reference-elastic buckling assembly calls this standard route.
        The one-key state is intentionally distinct from a committed nonlinear
        state, so current-state buckling still fails before mechanics.
        """

        del material
        if not isinstance(state, Mapping):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 current-state buckling is not authorized; "
                "reference buckling requires exactly one "
                "finite scalar key: axial_compression or axial_force"
            )
        keys = set(state)
        if keys == {"axial_compression"}:
            return self.compute_reference_geometric_stiffness(
                mesh,
                axial_compression=state["axial_compression"],
            )
        if keys == {"axial_force"}:
            return self.compute_reference_geometric_stiffness(
                mesh,
                axial_force=state["axial_force"],
            )
        raise GeBeam3IntegrationError(
            "qualified GE-B3 current-state buckling is not authorized; "
            "reference buckling requires exactly one "
            "finite scalar key: axial_compression or axial_force"
        )

    def compute_stresses(
        self,
        mesh: Any,
        displacements: Any,
        material: Any,
        return_global: bool = False,
    ) -> dict[str, Any]:
        del mesh, displacements, material, return_global
        raise GeBeam3IntegrationError(
            "qualified GE-B3 does not expose fibre stress without section authority; "
            "use recover_native_fields for generalized strains and resultants"
        )

    def _state_from_configuration(
        self,
        mesh: Any,
        total_u: Any,
        spatial_operators: Any,
    ) -> dict[str, Any]:
        total = np.asarray(total_u, dtype=np.float64)
        operators = np.asarray(spatial_operators, dtype=np.float64)
        if total.shape != (18,) or not np.all(np.isfinite(total)):
            raise GeBeam3CommittedStateError(
                "qualified GE-B3 committed coordinates must contain 18 finite values"
            )
        if operators.shape != (3, 3, 3) or not np.all(np.isfinite(operators)):
            raise GeBeam3CommittedStateError(
                "qualified GE-B3 committed spatial operators must have shape (3, 3, 3)"
            )
        authority = self._state_authority(mesh)
        frame = np.asarray(authority["reference_triad"], dtype=np.float64)
        absolute = np.einsum("nij,jk->nik", operators, frame)
        evaluation_u = np.array(total, copy=True).reshape(3, 6)
        evaluation_u[:, 3:] = 0.0
        _energy, _force, _tangent, fields = self.evaluate_candidate(
            mesh,
            evaluation_u.reshape(18),
            rotation_matrices=absolute,
            tangent=False,
        )
        return initialize_ge_beam3_state(
            **authority,
            committed_total_u=np.asarray(total, dtype="<f8"),
            committed_nodal_rotation_matrices=np.asarray(operators, dtype="<f8"),
            committed_local_rotation_matrices=np.asarray(
                fields["local_rotations"], dtype="<f8"
            ),
            committed_local_moments=np.asarray(
                fields["local_moments"], dtype="<f8"
            ),
            station_generalized_strain=np.asarray(
                fields["generalized_strain"], dtype="<f8"
            ),
            station_generalized_resultant=np.asarray(
                fields["generalized_resultant"], dtype="<f8"
            ),
        )

    def init_model_bound_nonlinear_state(
        self,
        mesh: Any,
        material: Any,
        num_layers: int,
    ) -> dict[str, Any]:
        del material
        if type(num_layers) is not int or num_layers <= 0:
            raise GeBeam3CommittedStateError(
                "qualified GE-B3 stateless generalized section requires a positive "
                "solver layer-count request"
            )
        identity = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)
        return self._state_from_configuration(mesh, np.zeros(18), identity)

    def validate_model_bound_nonlinear_state(
        self,
        mesh: Any,
        material: Any,
        state: Mapping[str, Any],
        num_layers: int,
        *,
        expected_committed_total_u: Any = None,
    ) -> dict[str, Any]:
        del material
        if type(num_layers) is not int or num_layers <= 0:
            raise GeBeam3CommittedStateError(
                "qualified GE-B3 stateless generalized section requires a positive "
                "solver layer-count request"
            )
        authority = self._state_authority(mesh)
        normalized = validate_ge_beam3_state(
            normalize_ge_beam3_checkpoint_state(state),
            **authority,
            expected_committed_total_u=expected_committed_total_u,
        )
        total = np.asarray(normalized["committed_total_u"], dtype=np.float64)
        frame = np.asarray(authority["reference_triad"], dtype=np.float64)
        operators = np.asarray(
            normalized["committed_nodal_rotation_matrices"], dtype=np.float64
        )
        absolute = np.einsum("nij,jk->nik", operators, frame)
        evaluation_u = np.array(total, copy=True).reshape(3, 6)
        evaluation_u[:, 3:] = 0.0
        _energy, _force, _tangent, fields = self.evaluate_candidate(
            mesh,
            evaluation_u.reshape(18),
            rotation_matrices=absolute,
            tangent=False,
        )
        return validate_ge_beam3_state(
            normalized,
            **authority,
            expected_committed_total_u=expected_committed_total_u,
            expected_local_state={
                "committed_local_rotation_matrices": fields["local_rotations"],
                "committed_local_moments": fields["local_moments"],
                "station_generalized_strain": fields["generalized_strain"],
                "station_generalized_resultant": fields["generalized_resultant"],
            },
        )

    def compute_nonlinear_response(
        self,
        mesh: Any,
        material: Any,
        displacement: Any,
        state: Optional[Mapping[str, Any]] = None,
        num_layers: int = 1,
        tangent: bool = True,
        *,
        native_rotation_trial: Optional[NativeElementRotationView] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
        """Evaluate one solver-owned finite static trial without committing it."""

        if native_rotation_trial is None:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 nonlinear mechanics requires an active "
                "NativeElementRotationView"
            )
        if type(num_layers) is not int or num_layers <= 0:
            raise GeBeam3CommittedStateError(
                "qualified GE-B3 stateless generalized section requires a positive "
                "solver layer-count request"
            )
        reference = self.get_node_coordinates(mesh)
        committed_total = np.empty((3, 6), dtype=np.float64)
        committed_total[:, :3] = (
            np.asarray(native_rotation_trial.committed_coordinates) - reference
        )
        committed_total[:, 3:] = np.asarray(
            native_rotation_trial.committed_rotation_coordinates
        )
        if state is None:
            committed = self._state_from_configuration(
                mesh,
                committed_total.reshape(18),
                native_rotation_trial.committed_rotation_matrices,
            )
        else:
            committed = self.validate_model_bound_nonlinear_state(
                mesh,
                material,
                state,
                num_layers,
            )
        recorded = np.asarray(committed["committed_total_u"], dtype=np.float64).reshape(3, 6)
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(native_rotation_trial.committed_coordinates))),
        )
        if not np.allclose(
            reference + recorded[:, :3],
            native_rotation_trial.committed_coordinates,
            rtol=0.0,
            atol=1.0e-12 * coordinate_scale,
        ):
            raise GeBeam3CommittedStateError(
                "committed state and node-shared coordinates disagree"
            )
        if not np.array_equal(
            recorded[:, 3:], native_rotation_trial.committed_rotation_coordinates
        ):
            raise GeBeam3CommittedStateError(
                "committed state and node-shared rotation coordinates disagree"
            )
        if not np.array_equal(
            committed["committed_nodal_rotation_matrices"],
            native_rotation_trial.committed_rotation_matrices,
        ):
            raise GeBeam3CommittedStateError(
                "committed state and node-shared spatial operators disagree"
            )

        _energy, force, matrix, fields, total = self._evaluate_solver_chart(
            mesh,
            displacement,
            native_rotation_trial,
            tangent=bool(tangent),
        )
        candidate = initialize_ge_beam3_state(
            **self._state_authority(mesh),
            committed_total_u=np.asarray(total, dtype="<f8"),
            committed_nodal_rotation_matrices=np.asarray(
                native_rotation_trial.trial_rotation_matrices, dtype="<f8"
            ),
            committed_local_rotation_matrices=np.asarray(
                fields["local_rotations"], dtype="<f8"
            ),
            committed_local_moments=np.asarray(
                fields["local_moments"], dtype="<f8"
            ),
            station_generalized_strain=np.asarray(
                fields["generalized_strain"], dtype="<f8"
            ),
            station_generalized_resultant=np.asarray(
                fields["generalized_resultant"], dtype="<f8"
            ),
        )
        self.validate_model_bound_nonlinear_state(
            mesh,
            material,
            candidate,
            num_layers,
            expected_committed_total_u=np.asarray(total, dtype=np.float64),
        )
        return (
            np.array(force, copy=True),
            None if matrix is None else np.array(matrix, copy=True),
            candidate,
        )

    def recover_native_fields(
        self,
        mesh: Any,
        displacement: Any,
        *,
        rotation_matrices: Any = None,
    ) -> dict[str, Any]:
        recovered = dict(
            super().recover_native_fields(
                mesh,
                displacement,
                rotation_matrices=rotation_matrices,
            )
        )
        recovered.pop("candidate_id", None)
        recovered["formulation_id"] = GE_BEAM3_QUALIFIED_FORMULATION_ID
        recovered["qualification_origin_candidate_id"] = (
            GE_BEAM3_MIXED_CANDIDATE_ID
        )
        provenance = dict(recovered["provenance"])
        provenance["formulation_schema"] = GE_BEAM3_FORMULATION_SCHEMA
        provenance["qualified_formulation_id"] = (
            GE_BEAM3_QUALIFIED_FORMULATION_ID
        )
        provenance["qualification_origin"] = dict(
            GE_BEAM3_QUALIFICATION_ORIGIN
        )
        provenance["state_layout_id"] = STATE_LAYOUT_ID
        recovered["provenance"] = provenance
        return recovered

    def to_dict(self, mesh: Any = None) -> dict[str, Any]:
        """Return the exact typed, JSON-compatible P3 element record.

        The reference geometry belongs to the model, so a newly constructed
        element must first be bound by passing ``mesh`` here or by evaluating
        any standard/native route once.
        """

        if type(self) is not GeometricallyExactBeam3D3NElement:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 serialization requires the exact public class"
            )
        if mesh is not None:
            self._reference_geometry(mesh)
        if (
            self._qualified_reference_geometry is None
            or self._qualified_reference_triad is None
        ):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 reference geometry must be model-bound before serialization"
            )
        section_stiffness = generalized_beam_stiffness(self.generalized_section)
        section_mass = generalized_beam_mass_matrix(self.generalized_section)
        assert section_mass is not None
        axis = self.reference_axis_direction
        if axis is None:
            axis = self._qualified_reference_triad[:, 0]
        section_descriptor = {
            "mass_matrix_per_reference_length": encode_typed_array(
                np.asarray(section_mass, dtype="<f8"),
                path="$.section_descriptor.mass_matrix_per_reference_length",
            ),
            "name": str(self.generalized_section.name),
            "schema": SECTION_SCHEMA,
            "stiffness_matrix": encode_typed_array(
                np.asarray(section_stiffness, dtype="<f8"),
                path="$.section_descriptor.stiffness_matrix",
            ),
        }
        return {
            "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
            "element_id": int(self.element_id),
            "formulation_id": GE_BEAM3_QUALIFIED_FORMULATION_ID,
            "formulation_schema": GE_BEAM3_FORMULATION_SCHEMA,
            "material_name": str(self.material_name),
            "node_ids": encode_typed_array(
                np.asarray(self.node_ids, dtype="<i8"), path="$.node_ids"
            ),
            "qualification_origin": dict(GE_BEAM3_QUALIFICATION_ORIGIN),
            "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
            "reference_axis_direction": encode_typed_array(
                np.asarray(axis, dtype="<f8"), path="$.reference_axis_direction"
            ),
            "reference_geometry": encode_typed_array(
                self._qualified_reference_geometry, path="$.reference_geometry"
            ),
            "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
            "reference_orientation": encode_typed_array(
                np.asarray(self.reference_orientation, dtype="<f8"),
                path="$.reference_orientation",
            ),
            "reference_triad": encode_typed_array(
                self._qualified_reference_triad, path="$.reference_triad"
            ),
            "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
            "section_descriptor": section_descriptor,
            "state_layout_id": STATE_LAYOUT_ID,
            "type": GE_BEAM3_SELECTOR,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "GeometricallyExactBeam3D3NElement":
        if cls is not GeometricallyExactBeam3D3NElement:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 deserialization requires the exact public class"
            )
        if not isinstance(payload, Mapping):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 element record must be a mapping"
            )
        data = dict(payload)
        if set(data) != _ELEMENT_KEYS:
            missing = sorted(_ELEMENT_KEYS - set(data))
            extra = sorted(set(data) - _ELEMENT_KEYS)
            raise GeBeam3IntegrationError(
                f"qualified GE-B3 element keys mismatch; missing={missing}, extra={extra}"
            )
        identities = {
            "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
            "formulation_id": GE_BEAM3_QUALIFIED_FORMULATION_ID,
            "formulation_schema": GE_BEAM3_FORMULATION_SCHEMA,
            "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
            "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
            "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
            "state_layout_id": STATE_LAYOUT_ID,
            "type": GE_BEAM3_SELECTOR,
        }
        for key, expected in identities.items():
            if type(data[key]) is not str or data[key] != expected:
                raise GeBeam3IntegrationError(
                    f"qualified GE-B3 serialized fingerprint {key} mismatch"
                )
        if data["qualification_origin"] != GE_BEAM3_QUALIFICATION_ORIGIN:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 qualification origin mismatch"
            )
        if not isinstance(data["qualification_origin"], Mapping) or set(
            data["qualification_origin"]
        ) != set(GE_BEAM3_QUALIFICATION_ORIGIN):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 qualification origin schema mismatch"
            )

        node_ids = decode_typed_array(
            data["node_ids"], dtype="<i8", shape=(3,), label="node_ids"
        )
        geometry = decode_typed_array(
            data["reference_geometry"],
            dtype="<f8",
            shape=(3, 3),
            label="reference_geometry",
        )
        triad = decode_typed_array(
            data["reference_triad"],
            dtype="<f8",
            shape=(3, 3),
            label="reference_triad",
        )
        orientation = decode_typed_array(
            data["reference_orientation"],
            dtype="<f8",
            shape=(3,),
            label="reference_orientation",
        )
        axis = decode_typed_array(
            data["reference_axis_direction"],
            dtype="<f8",
            shape=(3,),
            label="reference_axis_direction",
        )
        descriptor = data["section_descriptor"]
        if not isinstance(descriptor, Mapping) or set(descriptor) != _SECTION_KEYS:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 section descriptor keys mismatch"
            )
        if (
            type(descriptor["schema"]) is not str
            or descriptor["schema"] != SECTION_SCHEMA
        ):
            raise GeBeam3IntegrationError(
                "qualified GE-B3 section descriptor schema mismatch"
            )
        section_name = _exact_string(descriptor["name"], "section descriptor name")
        stiffness = decode_typed_array(
            descriptor["stiffness_matrix"],
            dtype="<f8",
            shape=(6, 6),
            label="section stiffness",
        )
        mass = decode_typed_array(
            descriptor["mass_matrix_per_reference_length"],
            dtype="<f8",
            shape=(6, 6),
            label="section mass",
        )
        made = cls(
            _exact_integer(data["element_id"], "element_id"),
            [int(value) for value in node_ids],
            _exact_string(data["material_name"], "material_name"),
            section=GeneralizedBeamSection(
                stiffness=stiffness,
                mass_matrix=mass,
                name=section_name,
            ),
            reference_orientation=orientation,
            reference_axis_direction=axis,
        )
        made._bind_reference_snapshot(geometry, triad, serialized=True)
        if made.to_dict() != data:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 element record is not a lossless canonical identity"
            )
        return made

    def to_bytes(self, mesh: Any = None) -> bytes:
        return serialize_ge_beam3_element(self, mesh=mesh)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "GeometricallyExactBeam3D3NElement":
        if cls is not GeometricallyExactBeam3D3NElement:
            raise GeBeam3IntegrationError(
                "qualified GE-B3 byte deserialization requires the exact public class"
            )
        return deserialize_ge_beam3_element(raw)

    @property
    def capability_gaps(self) -> frozenset[str]:
        return frozenset(
            {
                "beam_shell_connection",
                "contact_state",
                "conservative_follower_loads",
                "current_state_buckling",
                "current_state_modal",
                "current_state_modal_and_buckling",
                "curved_reference",
                "fibre_stress_without_section_authority",
                "finite_rotation_transient_dynamics",
                "gyroscopic_terms",
                "history_bearing_sections",
                "mixed_current_state_buckling",
                "mixed_current_state_modal",
                "nonconservative_follower_loads",
            }
        )


def serialize_ge_beam3_element(
    element: GeometricallyExactBeam3D3NElement,
    *,
    mesh: Any = None,
) -> bytes:
    """Serialize one exact qualified element record as canonical UTF-8 bytes."""

    if type(element) is not GeometricallyExactBeam3D3NElement:
        raise GeBeam3IntegrationError(
            "qualified GE-B3 serializer requires the exact public class"
        )
    return canonical_json_bytes(element.to_dict(mesh=mesh))


def deserialize_ge_beam3_element(raw: bytes) -> GeometricallyExactBeam3D3NElement:
    """Decode one strict canonical qualified element record."""

    decoded = strict_canonical_json_loads(raw)
    if not isinstance(decoded, Mapping):
        raise GeBeam3IntegrationError(
            "qualified GE-B3 element bytes must decode to a mapping"
        )
    made = GeometricallyExactBeam3D3NElement.from_dict(decoded)
    if serialize_ge_beam3_element(made) != raw:
        raise GeBeam3IntegrationError(
            "qualified GE-B3 element bytes do not round-trip canonically"
        )
    return made


__all__ = [
    "GE_BEAM3_FORMULATION_SCHEMA",
    "GE_BEAM3_QUALIFICATION_ORIGIN",
    "GE_BEAM3_QUALIFIED_FORMULATION_ID",
    "GE_BEAM3_SELECTOR",
    "GeBeam3CommittedStateError",
    "GeBeam3IntegrationError",
    "GeBeam3MixedCommittedStateError",
    "GeometricallyExactBeam3D3NElement",
    "deserialize_ge_beam3_element",
    "serialize_ge_beam3_element",
    "serialize_ge_beam3_state",
]
