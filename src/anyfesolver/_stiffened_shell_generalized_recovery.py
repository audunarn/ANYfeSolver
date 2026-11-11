"""Format cached generalized station updates; never evaluate constitutive mechanics.

The lifecycle owner binds the cached material response to the accepted endpoint and
its fixed predecessor. Saved accepted histories may differ from replay by the
owner's established tolerance and remain authoritative for plastic state.
"""
from dataclasses import dataclass
import json
from decimal import localcontext
from math import fsum, isfinite
import numpy as np

from ._ge_beam3_generalized_cell import GeneralizedCellHistory, POLICY as CELL_POLICY
from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator


POLICY = "CACHED_GENERALIZED_STATION_RESULTANTS_V1"


@dataclass(frozen=True)
class GeneralizedBeamStationRecovery:
    """Local section fields in native six-component work order.

    Strain is (three extension/shear strains, three curvatures in 1/m);
    resultants are (three forces in N, three moments in N m). Measure is
    reference arclength in m. Stored energy is the endpoint density in J/m;
    dissipation is the incremental density in J/m from the fixed predecessor.
    Neither field is an integrated or cumulative energy.
    Scalar pairs are (high, low); vector pairs retain separate owned arrays.
    No physical fibre stress distribution is supplied by this resultant law.
    """
    cell: int
    station: int
    xi: float
    measure: float
    local_strain: np.ndarray
    local_strain_low: np.ndarray
    local_resultants: np.ndarray
    local_resultants_low: np.ndarray
    reference_frame: np.ndarray
    current_frame: np.ndarray
    stored_energy_per_length: tuple
    dissipation_per_length: tuple
    plastic_increment: tuple
    plastic_strain: np.ndarray
    plastic_strain_low: np.ndarray
    accumulated: tuple
    branch: str


@dataclass(frozen=True)
class GeneralizedBeamElementRecovery:
    element_id: int
    stations: tuple
    recovery_policy: str = POLICY
    production_qualified: bool = False


def _owned(value, shape):
    try:
        a = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("finite generalized recovery array required") from error
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("finite exact-shape generalized recovery array required")
    a.setflags(write=False)
    return a


def _vector_pair(value):
    a = _owned(value, (2, 6))
    return _owned(a[0], (6,)), _owned(a[1], (6,))


def _scalar_pair(value):
    a = _owned(value, (2, 1))
    pair = (float(a[0, 0]), float(a[1, 0]))
    if not isfinite(fsum(pair)):
        raise ValueError("representable generalized recovery scalar required")
    return pair


def _frame(value):
    a = _owned(value, (3, 3))
    if np.linalg.norm(a.T @ a - np.eye(3)) > 1e-9 or abs(np.linalg.det(a)-1.) > 1e-9:
        raise ValueError("physical generalized recovery frame required")
    return a


def build(element, cells, material_bytes, accepted_history):
    """Detach accepted station fields from one cached operator material response.

    Identity/layout validation is performed here; the caller owns endpoint and
    accepted/predecessor consistency validation. This function never invokes
    operator.evaluate, operator.recover, or cell.response.
    """
    operator = element.operator
    if type(operator) is not RetainedGeneralizedOperator:
        raise ValueError("native retained generalized operator required")
    operator.guard()
    if (type(accepted_history) is not GeneralizedCellHistory or
            accepted_history.cell_identity != operator.cell.identity or
            len(accepted_history.stations) != len(operator.stations)):
        raise ValueError("complete cell-bound generalized accepted history required")
    # Paired native histories use the constitutive operator's 80-digit context.
    with localcontext() as context:
        context.prec=80
        operator.cell._origins(accepted_history)  # Validation only; no update.
    rotations = _owned(cells, (2, 3, 3))
    for rotation in rotations:
        _frame(rotation)
    if type(material_bytes) is not bytes:
        raise ValueError("cached generalized material bytes required")
    try:
        material = json.loads(material_bytes)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid cached generalized material response") from error
    if (type(material) is not dict or material.get("policy") != CELL_POLICY or
            material.get("cell_identity") != operator.cell.identity or
            type(material.get("stations")) is not list or
            len(material["stations"]) != len(operator.stations)):
        raise ValueError("matching complete generalized station response required")
    records = []
    for location, response, history in zip(operator.stations, material["stations"], accepted_history.stations):
        cell, station, xi, measure, reference = location
        if (type(cell) is not int or cell not in (0, 1) or
                type(station) is not int or station < 0 or
                not isfinite(xi) or not isfinite(measure) or measure <= 0.):
            raise ValueError("valid generalized reference station measure required")
        try:
            strain, strain_low = _vector_pair(response["strain"])
            resultants, resultants_low = _vector_pair(response["resultants"])
            stored = _scalar_pair(response["stored_energy"])
            dissipation = _scalar_pair(response["dissipation"])
            increment = _scalar_pair(response["plastic_increment"])
            branch = response["branch"]
        except (KeyError, TypeError) as error:
            raise ValueError("complete generalized station fields required") from error
        if branch not in ("ELASTIC", "PLASTIC", "YIELD_BOUNDARY") or any(
                fsum(pair) < 0. for pair in (stored, dissipation, increment)):
            raise ValueError("admissible generalized station material fields required")
        plastic = _owned(history.plastic, (6, 2))
        accumulated = tuple(float(v) for v in _owned(history.accumulated, (2,)))
        ref = _frame(reference)
        records.append(GeneralizedBeamStationRecovery(
            cell, station, float(xi), float(measure), strain, strain_low,
            resultants, resultants_low, ref, _frame(rotations[cell] @ ref),
            stored, dissipation, increment, _owned(plastic[:, 0], (6,)),
            _owned(plastic[:, 1], (6,)), accumulated, branch))
    return GeneralizedBeamElementRecovery(int(element.element_id), tuple(records))
