"""Internal traction-point Coulomb return map for a material surface frame.

The caller owns material-point correspondence and supplies the *relative*
increment in the current spatial frame. Frame columns are material tangent
directions followed by the physical normal. This local law has no public
contact admission until a surface adapter and solver transaction use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _vector3(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _frame3(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("material frame must be a finite 3 by 3 matrix")
    _validate_frame_bytes(result.tobytes())
    return result


@lru_cache(maxsize=128)
def _validate_frame_bytes(value):
    """Exact geometry-only validation reuse; no caller array is retained.

    Shape and finite checks precede every lookup. Changed frame bits always
    undergo the original orthonormality and polarity tests.
    """
    frame = np.frombuffer(value, dtype=np.float64).reshape(3, 3)
    if (not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=1.0e-9)
            or not np.isclose(np.linalg.det(frame), 1.0,
                              rtol=0.0, atol=1.0e-9)):
        raise ValueError("material frame must be right-handed and orthonormal")


def _frozen_vector(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


def _frozen_frame(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(_frozen_vector(row) for row in values)


@dataclass(frozen=True)
class FrictionPointState:
    """Committed physical history at one persistent material contact point."""

    material_frame: tuple[tuple[float, float, float], ...]
    elastic_slip: tuple[float, float, float]
    partner_key: str | None = None
    cumulative_plastic_slip: float = 0.0
    cumulative_dissipation_density: float = 0.0

    def __post_init__(self) -> None:
        frame = _frame3(self.material_frame)
        elastic = _vector3(self.elastic_slip, "elastic slip")
        if abs(float(elastic @ frame[:, 2])) > (
                1.0e-10 * float(np.linalg.norm(elastic)) + 1.0e-15):
            raise ValueError("elastic slip must be tangent to its material frame")
        if self.partner_key is not None and (
                not isinstance(self.partner_key, str) or not self.partner_key):
            raise ValueError("partner_key must be a nonempty string or None")
        if self.partner_key is None and np.any(elastic):
            raise ValueError("an open friction point cannot retain elastic slip")
        for name in ("cumulative_plastic_slip",
                     "cumulative_dissipation_density"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "material_frame", _frozen_frame(frame))
        object.__setattr__(self, "elastic_slip", _frozen_vector(elastic))


@dataclass(frozen=True)
class FrictionPointTrial:
    """Disposable response; commit ``candidate_state`` only with the increment."""

    branch: str
    traction: tuple[float, float, float]
    displacement_tangent: tuple[tuple[float, float, float], ...]
    pressure_derivative: tuple[float, float, float]
    candidate_state: FrictionPointState
    plastic_slip_increment: tuple[float, float, float]
    dissipated_increment_density: float
    stored_energy_density: float
    released_storage_density: float


def initial_friction_point(material_frame) -> FrictionPointState:
    """Zero-traction state, with material axes supplied by the geometry owner."""

    return FrictionPointState(
        _frozen_frame(_frame3(material_frame)), (0.0, 0.0, 0.0),
    )


def objective_relative_tangent_increment(
    *,
    source_committed,
    target_committed,
    source_candidate,
    target_candidate,
    target_frame_committed,
    target_frame_candidate,
) -> tuple[float, float, float]:
    """Relative slip increment for one persistent pair of physical points.

    The target material frame transports the *committed separation*, including
    its normal gap, to the candidate pose before the relative change is
    projected. A common rigid transformation therefore produces zero slip even
    while an open pair rotates. The caller must retain the same material
    points on both sides; projection changes and facet transfers need their
    own history map.
    """

    source_old = _vector3(source_committed, "committed source position")
    target_old = _vector3(target_committed, "committed target position")
    source_new = _vector3(source_candidate, "candidate source position")
    target_new = _vector3(target_candidate, "candidate target position")
    frame_old = _frame3(target_frame_committed)
    frame_new = _frame3(target_frame_candidate)
    rotation = frame_new @ frame_old.T
    previous_separation = source_old - target_old
    current_separation = source_new - target_new
    projector = frame_new[:, :2] @ frame_new[:, :2].T
    increment = projector @ (
        current_separation - rotation @ previous_separation
    )
    if not np.all(np.isfinite(increment)):
        raise ValueError("relative material-point motion exceeded finite range")
    return _frozen_vector(increment)


def objective_relative_tangent_variation(
    *,
    source_committed,
    target_committed,
    source_candidate,
    target_candidate,
    target_frame_committed,
    target_frame_candidate,
    source_position_variation,
    target_position_variation,
    target_frame_variation,
) -> tuple[float, float, float]:
    """Smooth candidate directional derivative of the fixed-pair increment."""

    old_relative = (_vector3(source_committed, "committed source position")
                    - _vector3(target_committed, "committed target position"))
    current_relative = (_vector3(source_candidate, "candidate source position")
                        - _vector3(target_candidate, "candidate target position"))
    old_frame = _frame3(target_frame_committed)
    frame = _frame3(target_frame_candidate)
    d_frame = np.asarray(target_frame_variation, dtype=float)
    if d_frame.shape != (3, 3) or not np.all(np.isfinite(d_frame)):
        raise ValueError("target frame variation must be a finite 3 by 3 matrix")
    d_relative = (
        _vector3(source_position_variation, "source position variation")
        - _vector3(target_position_variation, "target position variation")
    )
    tangent = frame[:, :2]
    d_tangent = d_frame[:, :2]
    projector = tangent @ tangent.T
    d_projector = d_tangent @ tangent.T + tangent @ d_tangent.T
    rotated_old = frame @ old_frame.T @ old_relative
    d_rotated_old = d_frame @ old_frame.T @ old_relative
    variation = (d_projector @ (current_relative - rotated_old)
                 + projector @ (d_relative - d_rotated_old))
    if not np.all(np.isfinite(variation)):
        raise ValueError("relative material-point variation exceeded finite range")
    return _frozen_vector(variation)


def trial_coulomb_point(
    committed: FrictionPointState,
    *,
    material_frame,
    relative_increment,
    normal_pressure: float,
    friction_coefficient: float,
    tangential_stiffness: float,
    partner_key: str | None,
) -> FrictionPointTrial:
    """Return the local stick/slip response without mutating committed state.

    Pressure and traction have units Pa, ``relative_increment`` and elastic
    slip have units m, and tangential stiffness has units Pa/m. Traction acts
    on the source and opposes positive source-relative tangential motion.
    Material-frame transport represents rigid rotation, including spin about
    the normal. A changed partner or opening starts later contact with zero
    elastic traction; accumulated slip/dissipation remains reportable.
    """

    if not isinstance(committed, FrictionPointState):
        raise TypeError("committed must be a FrictionPointState")
    frame = _frame3(material_frame)
    increment = _vector3(relative_increment, "relative increment")
    scalars = {}
    for name, value, positive in (
            ("normal_pressure", normal_pressure, False),
            ("friction_coefficient", friction_coefficient, False),
            ("tangential_stiffness", tangential_stiffness, True)):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a scalar") from error
        if not np.isfinite(scalar) or scalar < 0.0 or (positive and scalar == 0.0):
            raise ValueError(f"{name} must be finite and " +
                             ("positive" if positive else "nonnegative"))
        scalars[name] = scalar
    if scalars["normal_pressure"] > 0.0 and (
            not isinstance(partner_key, str) or not partner_key):
        raise ValueError("active friction requires a persistent partner_key")
    if partner_key is not None and (
            not isinstance(partner_key, str) or not partner_key):
        raise ValueError("partner_key must be a nonempty string or None")

    stiffness = scalars["tangential_stiffness"]
    pressure = scalars["normal_pressure"]
    coefficient = scalars["friction_coefficient"]
    prior_frame = _frame3(committed.material_frame)
    same_partner = (committed.partner_key is not None
                    and committed.partner_key == partner_key)
    prior_elastic = _vector3(committed.elastic_slip, "elastic slip")
    released = (0.0 if same_partner else
                0.5 * stiffness * float(prior_elastic @ prior_elastic))
    old = prior_elastic if same_partner else np.zeros(3)
    material_rotation = frame @ prior_frame.T
    transported = material_rotation @ old
    projector = frame[:, :2] @ frame[:, :2].T
    elastic_trial = projector @ (transported + increment)
    if not np.all(np.isfinite(elastic_trial)):
        raise ValueError("tangential trial slip exceeded finite range")
    zero = np.zeros(3)
    zero_matrix = np.zeros((3, 3))

    if pressure == 0.0:
        # Motion after opening stores no tangential regularization energy.
        released = 0.5 * stiffness * float(prior_elastic @ prior_elastic)
        state = FrictionPointState(
            _frozen_frame(frame), _frozen_vector(zero), None,
            committed.cumulative_plastic_slip,
            committed.cumulative_dissipation_density,
        )
        return FrictionPointTrial(
            "open", _frozen_vector(zero), _frozen_frame(zero_matrix),
            _frozen_vector(zero), state, _frozen_vector(zero),
            0.0, 0.0, released,
        )

    cap = coefficient * pressure
    magnitude = float(np.linalg.norm(elastic_trial))
    if not np.isfinite(cap) or not np.isfinite(magnitude):
        raise ValueError("friction trial exceeded finite range")
    if stiffness * magnitude <= cap:
        elastic = elastic_trial
        traction = -stiffness * elastic
        tangent = -stiffness * projector
        pressure_derivative = zero
        plastic = zero
        dissipated = 0.0
        branch = "stick"
    else:
        direction = elastic_trial / magnitude
        elastic = (cap / stiffness) * direction
        traction = -cap * direction
        tangent = -(cap / magnitude) * (
            projector - np.outer(direction, direction)
        )
        pressure_derivative = -coefficient * direction
        plastic = elastic_trial - elastic
        dissipated = cap * float(np.linalg.norm(plastic))
        branch = "slip"
    state = FrictionPointState(
        _frozen_frame(frame), _frozen_vector(elastic), partner_key,
        committed.cumulative_plastic_slip + float(np.linalg.norm(plastic)),
        committed.cumulative_dissipation_density + dissipated,
    )
    return FrictionPointTrial(
        branch, _frozen_vector(traction), _frozen_frame(tangent),
        _frozen_vector(pressure_derivative), state,
        _frozen_vector(plastic), dissipated,
        0.5 * stiffness * float(elastic @ elastic), released,
    )


def trial_coulomb_point_directional(
    committed: FrictionPointState,
    *,
    material_frame,
    material_frame_variation,
    relative_increment,
    relative_increment_variation,
    normal_pressure: float,
    pressure_variation: float,
    friction_coefficient: float,
    tangential_stiffness: float,
    partner_key: str | None,
    allow_stick_slip_equality: bool = False,
) -> tuple[float, float, float]:
    """Traction derivative with frame and pressure variation.

    Equality rejects by default. The explicit development option selects the
    branch returned by the projection (stick at exact equality), a member of
    the generalized derivative; it is not a smooth derivative at that kink.
    """

    response = trial_coulomb_point(
        committed, material_frame=material_frame,
        relative_increment=relative_increment,
        normal_pressure=normal_pressure,
        friction_coefficient=friction_coefficient,
        tangential_stiffness=tangential_stiffness,
        partner_key=partner_key,
    )
    frame = _frame3(material_frame)
    d_frame = np.asarray(material_frame_variation, dtype=float)
    if d_frame.shape != (3, 3) or not np.all(np.isfinite(d_frame)):
        raise ValueError("material frame variation must be a finite 3 by 3 matrix")
    increment = _vector3(relative_increment, "relative increment")
    d_increment = _vector3(relative_increment_variation,
                           "relative increment variation")
    try:
        d_pressure = float(pressure_variation)
    except (TypeError, ValueError) as error:
        raise ValueError("pressure variation must be a finite scalar") from error
    if not np.isfinite(d_pressure):
        raise ValueError("pressure variation must be a finite scalar")
    if float(friction_coefficient) == 0.0:
        return (0.0, 0.0, 0.0)
    if response.branch == "open":
        if d_pressure != 0.0:
            raise ValueError("normal activation has no smooth traction derivative")
        return (0.0, 0.0, 0.0)

    prior_frame = _frame3(committed.material_frame)
    same_partner = (committed.partner_key is not None
                    and committed.partner_key == partner_key)
    old_elastic = (_vector3(committed.elastic_slip, "elastic slip")
                   if same_partner else np.zeros(3))
    rotated_old = frame @ prior_frame.T @ old_elastic
    d_rotated_old = d_frame @ prior_frame.T @ old_elastic
    tangent = frame[:, :2]
    d_tangent = d_frame[:, :2]
    projector = tangent @ tangent.T
    d_projector = d_tangent @ tangent.T + tangent @ d_tangent.T
    trial_vector = projector @ (rotated_old + increment)
    d_trial_vector = (d_projector @ (rotated_old + increment)
                      + projector @ (d_rotated_old + d_increment))
    stiffness = float(tangential_stiffness)
    coefficient = float(friction_coefficient)
    cap = coefficient * float(normal_pressure)
    magnitude = float(np.linalg.norm(trial_vector))
    if not allow_stick_slip_equality and abs(stiffness*magnitude - cap) <= 1.0e-12 * max(
            1.0, stiffness*magnitude, cap):
        raise ValueError("stick/slip equality has no smooth traction derivative")
    if response.branch == "stick":
        derivative = -stiffness * d_trial_vector
    else:
        direction = trial_vector / magnitude
        derivative = (-coefficient * d_pressure * direction
                      - (cap / magnitude) *
                      ((np.eye(3) - np.outer(direction, direction))
                       @ d_trial_vector))
    if not np.all(np.isfinite(derivative)):
        raise ValueError("friction traction variation exceeded finite range")
    return _frozen_vector(derivative)
