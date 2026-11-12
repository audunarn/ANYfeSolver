"""Private two-cell mixed discrete-curvature GE-B3 candidate.

The element packages two published lowest-order mixed Simo--Reissner cells on
vertex pairs 1--2 and 2--3.  It is intentionally not exported or selectable.
All 18 public nodal coordinates remain external; the 12 material-moment and
six element-rotation coordinates are solved locally and Schur condensed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ._ge_beam3_mixed_ad import (
    Jet2,
    RotationDomainError,
    constant_matrix,
    dot,
    matmul,
    matvec,
    proper_rotation,
    rotation_exponential,
    rotation_log,
    so3_exp,
    so3_log,
    transpose,
)
from .beam_sections import (
    GeneralizedBeamSection,
    generalized_beam_mass_matrix,
    generalized_beam_stiffness,
    resolve_generalized_beam_section,
)
from .elements import Element
from ._native_rotation_state import NativeElementRotationView
from .ge_beam3_mixed_state import (
    GeBeam3MixedCommittedStateError,
    initialize_ge_beam3_mixed_state,
    validate_committed_ge_beam3_mixed_state,
)


GE_BEAM3_MIXED_CANDIDATE_ID = "CANDIDATE_GE_BEAM3_DC_MIXED_K1_MACRO_V2"
GE_BEAM3_MIXED_RESERVED_FORMULATION_ID = "GE_BEAM3_DC_MIXED_K1_MACRO_V2"
GE_BEAM3_MIXED_FORMULATION_ID = GE_BEAM3_MIXED_CANDIDATE_ID
GE_BEAM3_MIXED_SCHEMA_ID = "GE_BEAM3_MIXED_K1_MACRO_SCHEMA_V1"
GE_BEAM3_MIXED_QUADRATURE_ID = "K1_FORCE_GAUSS1_MOMENT_GAUSS2_V1"
GE_BEAM3_MIXED_ROTATION_ID = "ELEMENT_P0_VERTEX_HYBRID_SPATIAL_INCREMENT_V1"
GE_BEAM3_MIXED_CONDENSATION_ID = "TWO_CELL_18_LOCAL_SCHUR_V1"
GE_BEAM3_MIXED_REFERENCE_ID = "STRAIGHT_TWO_EQUAL_CELLS_V1"
GE_BEAM3_MIXED_RELATIVE_ROTATION_LIMIT = 0.9 * np.pi

_CELLS = ((0, 1), (1, 2))
_MOMENT_MASS = np.array(((1.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 1.0 / 3.0)))
_GEOMETRY_TOLERANCE = 1.0e-12


class GeBeam3MixedError(RuntimeError):
    """Base error for the private mixed candidate."""


class GeBeam3MixedGeometryError(GeBeam3MixedError, ValueError):
    """The straight two-cell reference geometry is inadmissible."""


class GeBeam3MixedLocalSolveError(GeBeam3MixedError):
    """The element-local mixed stationary problem did not converge."""


class GeBeam3MixedStateError(GeBeam3MixedError, ValueError):
    """A finite rotation was supplied outside the authoritative matrix path."""


@dataclass(frozen=True)
class _LocalState:
    rotations: np.ndarray
    moments: np.ndarray
    iterations: int
    residual_norm: float


def _value_matrix(matrix: Sequence[Sequence[Jet2]]) -> np.ndarray:
    return np.asarray([[entry.value for entry in row] for row in matrix])


def _value_vector(vector: Sequence[Jet2]) -> np.ndarray:
    return np.asarray([entry.value for entry in vector])


def _quadratic(vector: Sequence[Jet2], matrix: np.ndarray) -> Jet2:
    made = matvec(constant_matrix(matrix, vector[0].gradient.size), vector)
    return dot(vector, made)


def _geodesic_midpoint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = rotation_log(left.T @ right)
    return left @ rotation_exponential(0.5 * relative)


class GeometricallyExactBeam3D3NElement(Element):
    """Straight-reference, two-cell mixed GE-B3 candidate.

    This private class is a mechanics prototype.  It deliberately does not
    inherit from ``QuadraticBeamElement`` and is absent from public factories.
    """

    candidate_id = GE_BEAM3_MIXED_CANDIDATE_ID
    formulation_id = GE_BEAM3_MIXED_FORMULATION_ID
    formulation_native_total_lagrangian = True
    native_state_consistency_required = False
    nonlinear_material_response_mode = "stateless_fixed_generalized_section"
    solver_integration_authorized = False
    REVERSAL_STRAIN_MAP = np.diag((1.0, -1.0, 1.0, 1.0, -1.0, 1.0))

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
        nodes = [int(value) for value in node_ids]
        super().__init__(int(element_id), nodes, str(material_name))
        if len(nodes) != 3 or len(set(nodes)) != 3:
            raise ValueError("mixed GE-B3 requires three distinct nodes")
        self.cross_section = dict(cross_section or {})
        inline_orientation = self.cross_section.get("orientation")
        if reference_orientation is not None and inline_orientation is not None:
            raise ValueError("reference orientation must be supplied once")
        orientation = reference_orientation if reference_orientation is not None else inline_orientation
        values = np.asarray(orientation, dtype=np.float64) if orientation is not None else np.empty(0)
        if values.shape != (3,) or not np.all(np.isfinite(values)) or np.linalg.norm(values) == 0.0:
            raise GeBeam3MixedGeometryError("mixed GE-B3 requires a finite nonzero physical reference orientation")
        self.reference_orientation = np.array(values, copy=True)
        if reference_axis_direction is None:
            self.reference_axis_direction = None
        else:
            axis = np.asarray(reference_axis_direction, dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
                raise GeBeam3MixedGeometryError("reference_axis_direction must contain three finite nonzero values")
            self.reference_axis_direction = np.array(axis, copy=True)
        inline_section = self.cross_section.get(
            "generalized_section", self.cross_section.get("beam_section")
        )
        inline_stiffness = self.cross_section.get("generalized_stiffness")
        raw_section = (
            section
            if section is not None
            else inline_section
            if inline_section is not None
            else inline_stiffness
        )
        if (
            raw_section is not None
            and type(raw_section) is not GeneralizedBeamSection
            and callable(getattr(raw_section, "generalized_stiffness_matrix", None))
        ):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 P2 accepts only the immutable linear "
                "GeneralizedBeamSection; external mutable or history-bearing "
                "section protocols require a separately qualified trial/commit/"
                "discard contract"
            )
        resolved_section = resolve_generalized_beam_section(self.cross_section, section)
        if resolved_section is None:
            raise ValueError("mixed GE-B3 requires a symmetric positive-definite generalized 6x6 section")
        if type(resolved_section) is not GeneralizedBeamSection:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 P2 section must resolve to an immutable linear "
                "GeneralizedBeamSection"
            )
        stiffness = generalized_beam_stiffness(resolved_section)
        mass = generalized_beam_mass_matrix(resolved_section)
        self.generalized_section = GeneralizedBeamSection(
            stiffness=stiffness,
            mass_matrix=mass,
            name=str(getattr(resolved_section, "name", "")),
        )
        for private_input in (
            "beam_section",
            "generalized_mass_matrix",
            "generalized_mass_per_length",
            "generalized_section",
            "generalized_stiffness",
            "mass_per_length",
        ):
            self.cross_section.pop(private_input, None)
        transform = self.REVERSAL_STRAIN_MAP
        reversal_sensitive = not np.allclose(
            transform @ stiffness @ transform,
            stiffness,
            rtol=0.0,
            atol=1.0e-12 * max(1.0, float(np.linalg.norm(stiffness, ord=np.inf))),
        )
        if mass is not None:
            reversal_sensitive = reversal_sensitive or not np.allclose(
                transform @ mass @ transform,
                mass,
                rtol=0.0,
                atol=1.0e-12 * max(1.0, float(np.linalg.norm(mass, ord=np.inf))),
            )
        if reversal_sensitive and self.reference_axis_direction is None:
            raise GeBeam3MixedGeometryError(
                "reversal-sensitive coupled section stiffness or mass requires "
                "physical reference_axis_direction"
            )

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: Any) -> np.ndarray:
        coordinates = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise GeBeam3MixedGeometryError(f"mixed GE-B3 references missing node {node_id}")
            coordinates.append(np.asarray(node.coords(), dtype=np.float64))
        made = np.asarray(coordinates)
        if made.shape != (3, 3) or not np.all(np.isfinite(made)):
            raise GeBeam3MixedGeometryError("mixed GE-B3 coordinates must be finite")
        return made

    def _reference_geometry(self, mesh: Any) -> tuple[np.ndarray, float, np.ndarray, int]:
        coordinates = self.get_node_coordinates(mesh)
        chord = coordinates[2] - coordinates[0]
        length = float(np.linalg.norm(chord))
        if not np.isfinite(length) or length <= 1.0e-14:
            raise GeBeam3MixedGeometryError("mixed GE-B3 reference line has zero or nonfinite length")
        midpoint = 0.5 * (coordinates[0] + coordinates[2])
        if np.linalg.norm(coordinates[1] - midpoint) > _GEOMETRY_TOLERANCE * length:
            raise GeBeam3MixedGeometryError("mixed GE-B3 V2 requires two equal straight reference cells")
        e1 = chord / length
        projected = self.reference_orientation - float(self.reference_orientation @ e1) * e1
        if np.linalg.norm(projected) <= _GEOMETRY_TOLERANCE * np.linalg.norm(self.reference_orientation):
            raise GeBeam3MixedGeometryError("reference orientation is parallel to the beam axis")
        e2 = projected / np.linalg.norm(projected)
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)
        frame = np.column_stack((e1, e2, e3))
        polarity = 1
        if self.reference_axis_direction is not None:
            physical = self.reference_axis_direction / np.linalg.norm(self.reference_axis_direction)
            alignment = float(physical @ e1)
            if abs(abs(alignment) - 1.0) > _GEOMETRY_TOLERANCE:
                raise GeBeam3MixedGeometryError("reference_axis_direction must be parallel to the beam axis")
            polarity = 1 if alignment >= 0.0 else -1
        return coordinates, length, frame, polarity

    def reference_triad(self, mesh: Any) -> np.ndarray:
        return np.array(self._reference_geometry(mesh)[2], copy=True)

    def _section_stiffness(self, mesh: Any) -> np.ndarray:
        _coordinates, _length, _frame, polarity = self._reference_geometry(mesh)
        stiffness = generalized_beam_stiffness(self.generalized_section)
        if polarity < 0:
            transform = self.REVERSAL_STRAIN_MAP
            stiffness = transform @ stiffness @ transform
        return stiffness

    def _section_mass(self, mesh: Any) -> np.ndarray:
        _coordinates, _length, _frame, polarity = self._reference_geometry(mesh)
        mass = generalized_beam_mass_matrix(self.generalized_section)
        if mass is None:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 reference mass requires an explicit positive-definite "
                "generalized 6x6 inertia per unit reference length"
            )
        if polarity < 0:
            transform = self.REVERSAL_STRAIN_MAP
            mass = transform @ mass @ transform
        return mass

    def require_private_analysis_route(self, route: str) -> None:
        """Guard the preregistered private P2 capability boundary."""

        normalized = str(route).strip().upper()
        allowed = {
            "NATIVE_NONLINEAR_STATIC",
            "NATIVE_RECOVERY",
            "REFERENCE_BUCKLING",
            "REFERENCE_MODAL",
            "REFERENCE_STATIC",
        }
        if normalized not in allowed:
            raise GeBeam3MixedStateError(
                f"mixed GE-B3 route {normalized or '<empty>'} is outside the "
                "private P2 straight-reference authority"
            )

    @staticmethod
    def reverse_section_stiffness(stiffness: Any) -> np.ndarray:
        matrix = np.asarray(stiffness, dtype=np.float64)
        if matrix.shape != (6, 6):
            raise ValueError("section stiffness must have shape (6, 6)")
        transform = GeometricallyExactBeam3D3NElement.REVERSAL_STRAIN_MAP
        return transform @ matrix @ transform

    @staticmethod
    def _guard_vertex_rotations(rotations: np.ndarray) -> None:
        for left, right in _CELLS:
            angle = float(np.linalg.norm(rotation_log(rotations[left].T @ rotations[right])))
            if angle >= GE_BEAM3_MIXED_RELATIVE_ROTATION_LIMIT:
                raise RotationDomainError(
                    f"mixed GE-B3 relative vertex rotation {left + 1}-{right + 1} is {angle:.9g} rad; refine or cut back"
                )

    def _base_configuration(
        self,
        mesh: Any,
        displacement: Any,
        rotation_matrices: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        reference, length, frame, _polarity = self._reference_geometry(mesh)
        local = np.asarray(displacement, dtype=np.float64)
        if local.shape != (18,) or not np.all(np.isfinite(local)):
            raise ValueError("mixed GE-B3 displacement must contain 18 finite values")
        nodal = local.reshape(3, 6)
        positions = reference + nodal[:, :3]
        if np.any(nodal[:, 3:] != 0.0):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 rotation-vector coordinates are increments, not accumulated state; "
                "keep them zero and supply authoritative rotation_matrices"
            )
        if rotation_matrices is None:
            rotations = np.repeat(frame[None, :, :], 3, axis=0)
        else:
            supplied = np.asarray(rotation_matrices, dtype=np.float64)
            if supplied.shape != (3, 3, 3):
                raise ValueError("rotation_matrices must have shape (3, 3, 3)")
            rotations = np.asarray([proper_rotation(value, label=f"rotation_matrices[{index}]") for index, value in enumerate(supplied)])
        self._guard_vertex_rotations(rotations)
        return positions, rotations, 0.5 * length, frame, self._section_stiffness(mesh)

    def _solver_chart_configuration(
        self,
        mesh: Any,
        displacement: Any,
        native_rotation_trial: NativeElementRotationView,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Validate and materialize one solver-owned multiplicative trial."""

        if type(native_rotation_trial) is not NativeElementRotationView:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 nonlinear mechanics requires an exact "
                "NativeElementRotationView"
            )
        if native_rotation_trial.trial_serial is None:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 nonlinear mechanics requires an active trial"
            )
        if int(native_rotation_trial.element_id) != int(self.element_id):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native rotation view has the wrong element ID"
            )
        if tuple(native_rotation_trial.node_ids) != tuple(self.node_ids):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native rotation view has the wrong connectivity"
            )
        total = np.asarray(displacement, dtype=np.float64)
        if total.shape != (18,) or not np.all(np.isfinite(total)):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 solver displacement must contain 18 finite values"
            )
        by_node = total.reshape(3, 6)
        reference, length, frame, _polarity = self._reference_geometry(mesh)
        trial_coordinates = np.asarray(
            native_rotation_trial.trial_coordinates, dtype=np.float64
        )
        if trial_coordinates.shape != (3, 3) or not np.all(np.isfinite(trial_coordinates)):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native trial coordinates are malformed"
            )
        scale = max(1.0, float(length), float(np.linalg.norm(trial_coordinates, ord=np.inf)))
        if not np.allclose(
            trial_coordinates,
            reference + by_node[:, :3],
            rtol=0.0,
            atol=1.0e-12 * scale,
        ):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 displacement and node-shared trial coordinates disagree"
            )
        trial_rotation_coordinates = np.asarray(
            native_rotation_trial.trial_rotation_coordinates, dtype=np.float64
        )
        if not np.array_equal(by_node[:, 3:], trial_rotation_coordinates):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 displacement and node-shared rotation coordinates disagree"
            )
        committed_operators = np.asarray(
            native_rotation_trial.committed_rotation_matrices, dtype=np.float64
        )
        trial_operators = np.asarray(
            native_rotation_trial.trial_rotation_matrices, dtype=np.float64
        )
        increments = np.asarray(
            native_rotation_trial.rotation_coordinate_increment, dtype=np.float64
        )
        if (
            committed_operators.shape != (3, 3, 3)
            or trial_operators.shape != (3, 3, 3)
            or increments.shape != (3, 3)
            or not np.all(np.isfinite(increments))
        ):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native rotation trial is malformed"
            )
        committed_absolute = np.einsum("nij,jk->nik", committed_operators, frame)
        trial_absolute = np.einsum("nij,jk->nik", trial_operators, frame)
        reconstructed = np.asarray(
            [
                rotation_exponential(increments[node]) @ committed_absolute[node]
                for node in range(3)
            ]
        )
        if not np.allclose(reconstructed, trial_absolute, rtol=0.0, atol=2.0e-12):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native operators do not reproduce the solver chart"
            )
        self._guard_vertex_rotations(trial_absolute)
        return (
            np.array(trial_coordinates, copy=True),
            np.array(trial_absolute, copy=True),
            0.5 * float(length),
            np.array(frame, copy=True),
            self._section_stiffness(mesh),
            np.array(committed_absolute, copy=True),
            np.array(increments, copy=True),
            np.array(total, copy=True),
        )

    @staticmethod
    def _potential(
        positions: np.ndarray,
        vertex_rotations: np.ndarray,
        cell_length: float,
        section: np.ndarray,
        local_rotations: np.ndarray,
        moments: np.ndarray,
        *,
        include_external: bool,
        chart_rotation_increment: Optional[np.ndarray] = None,
        chart_committed_rotations: Optional[np.ndarray] = None,
    ) -> Jet2:
        external_size = 18 if include_external else 0
        size = external_size + 18
        variables = [Jet2.variable(0.0, index, size) for index in range(size)]
        made_positions: list[list[Jet2]] = []
        made_vertices: list[list[list[Jet2]]] = []
        for node in range(3):
            made_positions.append(
                [
                    Jet2.constant(positions[node, axis], size)
                    + (variables[node * 6 + axis] if include_external else 0.0)
                    for axis in range(3)
                ]
            )
            if include_external:
                if (chart_rotation_increment is None) != (
                    chart_committed_rotations is None
                ):
                    raise ValueError(
                        "mixed GE-B3 solver chart requires both rotation "
                        "increments and committed frames"
                    )
                if chart_rotation_increment is None:
                    rotation_coordinates = variables[node * 6 + 3 : node * 6 + 6]
                    rotation_base = vertex_rotations[node]
                else:
                    increments = np.asarray(chart_rotation_increment, dtype=np.float64)
                    bases = np.asarray(chart_committed_rotations, dtype=np.float64)
                    if increments.shape != (3, 3) or bases.shape != (3, 3, 3):
                        raise ValueError(
                            "mixed GE-B3 solver-chart rotations have incompatible shape"
                        )
                    rotation_coordinates = [
                        Jet2.constant(float(increments[node, axis]), size)
                        + variables[node * 6 + 3 + axis]
                        for axis in range(3)
                    ]
                    rotation_base = bases[node]
                increment = so3_exp(rotation_coordinates)
                made_vertices.append(
                    matmul(increment, constant_matrix(rotation_base, size))
                )
            else:
                made_vertices.append(constant_matrix(vertex_rotations[node], size))
        a = section[:3, :3]
        b = section[:3, 3:]
        d_inverse = np.linalg.inv(section[3:, 3:])
        total = Jet2.constant(0.0, size)
        e1 = [Jet2.constant(1.0, size), Jet2.constant(0.0, size), Jet2.constant(0.0, size)]
        for cell, (left_node, right_node) in enumerate(_CELLS):
            start = external_size + cell * 9
            local_increment = so3_exp(variables[start : start + 3])
            local_rotation = matmul(local_increment, constant_matrix(local_rotations[cell], size))
            cell_moments: list[list[Jet2]] = []
            for endpoint in range(2):
                moment_start = start + 3 + endpoint * 3
                cell_moments.append(
                    [Jet2.constant(moments[cell, endpoint, component], size) + variables[moment_start + component] for component in range(3)]
                )
            tangent = [
                (made_positions[right_node][axis] - made_positions[left_node][axis]) / cell_length
                for axis in range(3)
            ]
            gamma = [entry - ref for entry, ref in zip(matvec(transpose(local_rotation), tangent), e1)]
            total = total + 0.5 * cell_length * _quadratic(gamma, a)
            bt_gamma = matvec(constant_matrix(b.T, size), gamma)
            complementary = [[entry - coupled for entry, coupled in zip(endpoint, bt_gamma)] for endpoint in cell_moments]
            for row in range(2):
                for column in range(2):
                    total = total - 0.5 * cell_length * _MOMENT_MASS[row, column] * dot(
                        complementary[row],
                        matvec(constant_matrix(d_inverse, size), complementary[column]),
                    )
            relative_left = so3_log(matmul(transpose(local_rotation), made_vertices[left_node]))
            relative_right = so3_log(matmul(transpose(local_rotation), made_vertices[right_node]))
            total = total + dot(relative_right, cell_moments[1]) - dot(relative_left, cell_moments[0])
        return total

    @classmethod
    def _solve_local(
        cls,
        positions: np.ndarray,
        vertex_rotations: np.ndarray,
        cell_length: float,
        section: np.ndarray,
    ) -> _LocalState:
        local_rotations = np.asarray(
            [_geodesic_midpoint(vertex_rotations[left], vertex_rotations[right]) for left, right in _CELLS]
        )
        moments = np.zeros((2, 2, 3))
        initial_norm = None
        for iteration in range(30):
            potential = cls._potential(
                positions,
                vertex_rotations,
                cell_length,
                section,
                local_rotations,
                moments,
                include_external=False,
            )
            residual = potential.gradient
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if initial_norm is None:
                initial_norm = max(1.0, residual_norm)
            if residual_norm <= 2.0e-11 * initial_norm:
                return _LocalState(np.array(local_rotations, copy=True), np.array(moments, copy=True), iteration, residual_norm)
            hessian = 0.5 * (potential.hessian + potential.hessian.T)
            try:
                step = np.linalg.solve(hessian, -residual)
            except np.linalg.LinAlgError as exc:
                raise GeBeam3MixedLocalSolveError("mixed GE-B3 local Hessian is singular") from exc
            accepted = False
            for exponent in range(12):
                scale = 0.5**exponent
                trial_rotations = np.asarray(
                    [rotation_exponential(scale * step[cell * 9 : cell * 9 + 3]) @ local_rotations[cell] for cell in range(2)]
                )
                trial_moments = np.array(moments, copy=True)
                for cell in range(2):
                    trial_moments[cell, 0] += scale * step[cell * 9 + 3 : cell * 9 + 6]
                    trial_moments[cell, 1] += scale * step[cell * 9 + 6 : cell * 9 + 9]
                trial = cls._potential(
                    positions,
                    vertex_rotations,
                    cell_length,
                    section,
                    trial_rotations,
                    trial_moments,
                    include_external=False,
                )
                if np.linalg.norm(trial.gradient, ord=np.inf) < residual_norm:
                    local_rotations, moments = trial_rotations, trial_moments
                    accepted = True
                    break
            if not accepted:
                raise GeBeam3MixedLocalSolveError("mixed GE-B3 local Newton line search stalled")
        raise GeBeam3MixedLocalSolveError("mixed GE-B3 local Newton exceeded 30 iterations")

    @staticmethod
    def _recover_section_fields(
        positions: np.ndarray,
        cell_length: float,
        section: np.ndarray,
        local: _LocalState,
    ) -> tuple[np.ndarray, np.ndarray]:
        strains: list[np.ndarray] = []
        resultants: list[np.ndarray] = []
        d_inverse = np.linalg.inv(section[3:, 3:])
        for cell, (left, right) in enumerate(_CELLS):
            gamma = (
                local.rotations[cell].T
                @ ((positions[right] - positions[left]) / cell_length)
                - np.array((1.0, 0.0, 0.0))
            )
            for endpoint in range(2):
                moment = local.moments[cell, endpoint]
                curvature = d_inverse @ (
                    moment - section[:3, 3:].T @ gamma
                )
                strain = np.concatenate((gamma, curvature))
                strains.append(strain)
                resultants.append(section @ strain)
        return np.asarray(strains), np.asarray(resultants)

    @classmethod
    def _candidate_fields(
        cls,
        positions: np.ndarray,
        cell_length: float,
        frame: np.ndarray,
        section: np.ndarray,
        local: _LocalState,
    ) -> dict[str, Any]:
        strains, resultants = cls._recover_section_fields(
            positions, cell_length, section, local
        )
        return {
            "candidate_id": GE_BEAM3_MIXED_CANDIDATE_ID,
            "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
            "formulation_id": GE_BEAM3_MIXED_FORMULATION_ID,
            "generalized_resultant": resultants,
            "generalized_strain": strains,
            "local_iterations": local.iterations,
            "local_moments": np.array(local.moments, copy=True),
            "local_residual_norm": local.residual_norm,
            "local_rotations": np.array(local.rotations, copy=True),
            "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
            "reference_frame": np.array(frame, copy=True),
            "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
            "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
            "schema_id": GE_BEAM3_MIXED_SCHEMA_ID,
        }

    def evaluate_candidate(
        self,
        mesh: Any,
        displacement: Any,
        *,
        rotation_matrices: Any = None,
        tangent: bool = True,
    ) -> tuple[float, np.ndarray, Optional[np.ndarray], dict[str, Any]]:
        positions, rotations, cell_length, frame, section = self._base_configuration(
            mesh, displacement, rotation_matrices
        )
        local = self._solve_local(positions, rotations, cell_length, section)
        potential = self._potential(
            positions,
            rotations,
            cell_length,
            section,
            local.rotations,
            local.moments,
            include_external=True,
        )
        residual = np.array(potential.gradient[:18], copy=True)
        made_tangent = None
        if tangent:
            hessian = 0.5 * (potential.hessian + potential.hessian.T)
            hee = hessian[:18, :18]
            hei = hessian[:18, 18:]
            hii = hessian[18:, 18:]
            made_tangent = hee - hei @ np.linalg.solve(hii, hei.T)
            made_tangent = 0.5 * (made_tangent + made_tangent.T)
        fields = self._candidate_fields(
            positions, cell_length, frame, section, local
        )
        return float(potential.value), residual, made_tangent, fields

    def _evaluate_solver_chart(
        self,
        mesh: Any,
        displacement: Any,
        native_rotation_trial: NativeElementRotationView,
        *,
        tangent: bool,
    ) -> tuple[float, np.ndarray, Optional[np.ndarray], dict[str, Any], np.ndarray]:
        (
            positions,
            trial_rotations,
            cell_length,
            frame,
            section,
            committed_rotations,
            rotation_increment,
            total,
        ) = self._solver_chart_configuration(
            mesh, displacement, native_rotation_trial
        )
        local = self._solve_local(
            positions, trial_rotations, cell_length, section
        )
        potential = self._potential(
            positions,
            trial_rotations,
            cell_length,
            section,
            local.rotations,
            local.moments,
            include_external=True,
            chart_rotation_increment=rotation_increment,
            chart_committed_rotations=committed_rotations,
        )
        residual = np.array(potential.gradient[:18], copy=True)
        made_tangent = None
        if tangent:
            hessian = 0.5 * (potential.hessian + potential.hessian.T)
            external_internal = hessian[:18, 18:]
            try:
                made_tangent = hessian[:18, :18] - external_internal @ np.linalg.solve(
                    hessian[18:, 18:], external_internal.T
                )
            except np.linalg.LinAlgError as exc:
                raise GeBeam3MixedLocalSolveError(
                    "mixed GE-B3 solver-chart local Hessian is singular"
                ) from exc
            made_tangent = 0.5 * (made_tangent + made_tangent.T)
        fields = self._candidate_fields(
            positions, cell_length, frame, section, local
        )
        fields["solver_chart_id"] = (
            "EXP_DELTA_PLUS_X_LEFT_MULTIPLIES_COMMITTED_OPERATOR_V1"
        )
        fields["spatial_operator_rotation_increment"] = np.array(
            rotation_increment, copy=True
        )
        return float(potential.value), residual, made_tangent, fields, total

    def compute_candidate_stiffness_matrix(self, mesh: Any) -> np.ndarray:
        """Return the research-gate reference tangent without enabling assembly."""

        _energy, _force, tangent, _fields = self.evaluate_candidate(
            mesh, np.zeros(18), tangent=True
        )
        assert tangent is not None
        return np.array(tangent, copy=True)

    @staticmethod
    def _dead_load_rows(value: Any, label: str) -> np.ndarray:
        rows = np.asarray(value, dtype=np.float64)
        if rows.shape == (3,):
            rows = np.repeat(rows[None, :], 3, axis=0)
        if rows.shape != (3, 3) or not np.all(np.isfinite(rows)):
            raise GeBeam3MixedStateError(
                f"mixed GE-B3 {label} must contain one or three finite 3-vectors"
            )
        return np.array(rows, copy=True)

    def compute_reference_line_load(
        self,
        mesh: Any,
        descriptor: Mapping[str, Any],
    ) -> np.ndarray:
        """Return a private, reference-arclength consistent dead-load vector."""

        if not isinstance(descriptor, Mapping):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 reference-line load descriptor must be a mapping"
            )
        classification = str(descriptor.get("classification", "")).upper()
        if classification not in {"SPATIAL_DEAD", "MATERIAL_DEAD"}:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 P2 supports only spatial-dead or material-dead "
                "reference-line loads"
            )
        exact_keys = {
            "classification",
            "couple_per_reference_length_at_nodes",
            "force_per_reference_length_at_nodes",
        }
        if set(descriptor) != exact_keys:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 reference-line load descriptor has missing or "
                "unknown keys"
            )
        forces = self._dead_load_rows(
            descriptor["force_per_reference_length_at_nodes"], "line force"
        )
        couples = self._dead_load_rows(
            descriptor["couple_per_reference_length_at_nodes"], "line couple"
        )
        _coordinates, length, frame, _polarity = self._reference_geometry(mesh)
        if classification == "MATERIAL_DEAD":
            forces = np.einsum("ij,nj->ni", frame, forces)
            couples = np.einsum("ij,nj->ni", frame, couples)
        made = np.zeros((3, 6), dtype=np.float64)
        cell_length = 0.5 * length
        for left, right in _CELLS:
            made[left, :3] += cell_length * (2.0 * forces[left] + forces[right]) / 6.0
            made[right, :3] += cell_length * (forces[left] + 2.0 * forces[right]) / 6.0
            made[left, 3:] += cell_length * (2.0 * couples[left] + couples[right]) / 6.0
            made[right, 3:] += cell_length * (couples[left] + 2.0 * couples[right]) / 6.0
        return made.reshape(18)

    def compute_mass_matrix(self, mesh: Any, material: Any = None) -> np.ndarray:
        """Return reference-configuration consistent generalized inertia."""

        del material
        _coordinates, length, frame, _polarity = self._reference_geometry(mesh)
        local_mass = self._section_mass(mesh)
        rotation = np.zeros((6, 6), dtype=np.float64)
        rotation[:3, :3] = frame
        rotation[3:, 3:] = frame
        spatial_mass = rotation @ local_mass @ rotation.T
        made = np.zeros((18, 18), dtype=np.float64)
        cell_length = 0.5 * length
        for left, right in _CELLS:
            for row_node, row_weight in ((left, 0), (right, 1)):
                for column_node, column_weight in ((left, 0), (right, 1)):
                    coefficient = cell_length * _MOMENT_MASS[row_weight, column_weight]
                    rows = slice(6 * row_node, 6 * row_node + 6)
                    columns = slice(6 * column_node, 6 * column_node + 6)
                    made[rows, columns] += coefficient * spatial_mass
        return 0.5 * (made + made.T)

    def compute_reference_geometric_stiffness(
        self,
        mesh: Any,
        *,
        axial_compression: Any = None,
        axial_force: Any = None,
    ) -> np.ndarray:
        """Return the frozen compression-positive reference Euler operator."""

        if (axial_compression is None) == (axial_force is None):
            raise GeBeam3MixedStateError(
                "mixed GE-B3 geometric stiffness requires exactly one of "
                "axial_compression or axial_force"
            )
        raw = axial_compression if axial_compression is not None else axial_force
        if isinstance(raw, (bool, np.bool_)):
            raise GeBeam3MixedStateError("mixed GE-B3 axial force must be a real scalar")
        try:
            compression = float(raw)
        except (TypeError, ValueError) as exc:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 axial force must be a real scalar"
            ) from exc
        if axial_force is not None:
            compression = -compression
        if not np.isfinite(compression):
            raise GeBeam3MixedStateError("mixed GE-B3 axial force must be finite")
        _coordinates, length, frame, _polarity = self._reference_geometry(mesh)
        transverse = frame[:, 1:] @ frame[:, 1:].T
        cell_length = 0.5 * length
        made = np.zeros((18, 18), dtype=np.float64)
        for left, right in _CELLS:
            scaled = compression / cell_length * transverse
            left_rows = slice(6 * left, 6 * left + 3)
            right_rows = slice(6 * right, 6 * right + 3)
            made[left_rows, left_rows] += scaled
            made[right_rows, right_rows] += scaled
            made[left_rows, right_rows] -= scaled
            made[right_rows, left_rows] -= scaled
        return 0.5 * (made + made.T)

    def recover_native_fields(
        self,
        mesh: Any,
        displacement: Any,
        *,
        rotation_matrices: Any = None,
    ) -> dict[str, Any]:
        """Recover native four-sided station data without legacy beam paths."""

        energy, _force, _tangent, fields = self.evaluate_candidate(
            mesh,
            displacement,
            rotation_matrices=rotation_matrices,
            tangent=False,
        )
        current_frames = np.repeat(
            np.asarray(fields["local_rotations"], dtype=np.float64), 2, axis=0
        )
        local_resultants = np.asarray(
            fields["generalized_resultant"], dtype=np.float64
        )
        global_resultants = np.empty_like(local_resultants)
        for station in range(4):
            global_resultants[station, :3] = (
                current_frames[station] @ local_resultants[station, :3]
            )
            global_resultants[station, 3:] = (
                current_frames[station] @ local_resultants[station, 3:]
            )
        return {
            "candidate_id": GE_BEAM3_MIXED_CANDIDATE_ID,
            "energy": float(energy),
            "fibre_stress_available": False,
            "formulation_id": GE_BEAM3_MIXED_FORMULATION_ID,
            "global_generalized_resultant": global_resultants,
            "local_generalized_resultant": np.array(local_resultants, copy=True),
            "local_generalized_strain": np.array(
                fields["generalized_strain"], copy=True
            ),
            "provenance": {
                "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
                "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
                "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
                "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
                "schema_id": GE_BEAM3_MIXED_SCHEMA_ID,
            },
            "reference_frames": np.repeat(
                np.asarray(fields["reference_frame"])[None, :, :], 4, axis=0
            ),
            "station_frames": current_frames,
            "station_order": (
                "XI_MINUS_1",
                "XI_ZERO_LEFT",
                "XI_ZERO_RIGHT",
                "XI_PLUS_1",
            ),
        }

    def native_reference_directors(self, mesh: Any) -> np.ndarray:
        """Return the element-owned material-two direction at each vertex.

        The native rotation store owns one spatial operator per mesh node.  The
        reference director remains element-owned so two beams meeting at a node
        can retain distinct material rolls while sharing the same operator.
        """

        frame = self.reference_triad(mesh)
        return np.repeat(frame[None, :, 1], 3, axis=0)

    def _state_authority(self, mesh: Any) -> dict[str, Any]:
        reference, _length, frame, _polarity = self._reference_geometry(mesh)
        axis = self.reference_axis_direction
        if axis is None:
            axis = frame[:, 0]
        return {
            "element_id": int(self.element_id),
            "node_ids": np.asarray(self.node_ids, dtype="<i8"),
            "reference_geometry": np.asarray(reference, dtype="<f8"),
            "reference_triad": np.asarray(frame, dtype="<f8"),
            "reference_orientation": np.asarray(
                self.reference_orientation, dtype="<f8"
            ),
            "reference_axis_direction": np.asarray(axis, dtype="<f8"),
            "section_name": str(self.generalized_section.name),
            "section_stiffness": np.asarray(
                self._section_stiffness(mesh), dtype="<f8"
            ),
            "section_mass": np.asarray(self._section_mass(mesh), dtype="<f8"),
        }

    def _state_from_configuration(
        self,
        mesh: Any,
        total_u: Any,
        spatial_operators: Any,
    ) -> dict[str, Any]:
        total = np.asarray(total_u, dtype=np.float64)
        operators = np.asarray(spatial_operators, dtype=np.float64)
        if total.shape != (18,) or not np.all(np.isfinite(total)):
            raise GeBeam3MixedCommittedStateError(
                "mixed GE-B3 committed solver coordinates must contain 18 "
                "finite values"
            )
        if operators.shape != (3, 3, 3):
            raise GeBeam3MixedCommittedStateError(
                "mixed GE-B3 committed spatial operators must have shape "
                "(3, 3, 3)"
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
        return initialize_ge_beam3_mixed_state(
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
        """Create the exact zero-configuration P2 committed state."""

        del material
        if type(num_layers) is not int or num_layers != 1:
            raise GeBeam3MixedCommittedStateError(
                "mixed GE-B3 stateless generalized section requires one layer"
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
        """Validate state identity, integrity, and a native mechanics replay."""

        del material
        if type(num_layers) is not int or num_layers != 1:
            raise GeBeam3MixedCommittedStateError(
                "mixed GE-B3 stateless generalized section requires one layer"
            )
        authority = self._state_authority(mesh)
        normalized = validate_committed_ge_beam3_mixed_state(
            state,
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
        return validate_committed_ge_beam3_mixed_state(
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

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        del mesh, material
        raise GeBeam3MixedStateError(
            "mixed GE-B3 linear solver integration is not authorized; "
            "use the private compute_candidate_stiffness_matrix research gate only"
        )

    def compute_internal_forces(self, mesh: Any, displacements: Any, material: Any) -> np.ndarray:
        del mesh, displacements, material
        raise GeBeam3MixedStateError(
            "mixed GE-B3 solver integration is not authorized; use the private evaluate_candidate gate only"
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
        """Evaluate one solver-owned P2 static trial without committing it."""

        if native_rotation_trial is None:
            raise GeBeam3MixedStateError(
                "mixed GE-B3 native rotation transaction is not implemented "
                "without an active NativeElementRotationView"
            )
        if type(num_layers) is not int or num_layers != 1:
            raise GeBeam3MixedCommittedStateError(
                "mixed GE-B3 stateless generalized section requires one layer"
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
            recorded = np.asarray(
                committed["committed_total_u"], dtype=np.float64
            ).reshape(3, 6)
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
                raise GeBeam3MixedCommittedStateError(
                    "committed state and node-shared coordinates disagree"
                )
            if not np.array_equal(
                recorded[:, 3:],
                native_rotation_trial.committed_rotation_coordinates,
            ):
                raise GeBeam3MixedCommittedStateError(
                    "committed state and node-shared rotation coordinates disagree"
                )
            if not np.array_equal(
                committed["committed_nodal_rotation_matrices"],
                native_rotation_trial.committed_rotation_matrices,
            ):
                raise GeBeam3MixedCommittedStateError(
                    "committed state and node-shared spatial operators disagree"
                )
        _energy, force, matrix, fields, total = self._evaluate_solver_chart(
            mesh,
            displacement,
            native_rotation_trial,
            tangent=bool(tangent),
        )
        candidate = initialize_ge_beam3_mixed_state(
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
        return np.array(force, copy=True), (
            None if matrix is None else np.array(matrix, copy=True)
        ), candidate

    @property
    def capability_gaps(self) -> frozenset[str]:
        return frozenset(
            {
                "beam_shell_connection",
                "buckling",
                "curved_reference",
                "distributed_follower_loads",
                "history_bearing_sections",
                "finite_rotation_transient_dynamics",
                "current_state_modal_and_buckling",
                "public_selector",
            }
        )
