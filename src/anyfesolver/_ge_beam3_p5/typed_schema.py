"""Internal P5 package candidate; unqualified and not publicly registered.

Extracted mechanically from the bound source map. Do not edit copied mechanics
without a reviewed successor mapping and renewed equivalence checks.
"""

from dataclasses import dataclass
import math
import numpy as np
from .section import SectionHistory, SectionResponse, _history
from .mixed import BeamStationTrial, NonlinearCondensedResponse

@dataclass(frozen=True)
class NonlinearPathState:
    epoch: int
    positions: np.ndarray
    frames: np.ndarray
    forces: np.ndarray
    histories: tuple


@dataclass(frozen=True)
class NonlinearPathTrial:
    origin_epoch: int
    positions: np.ndarray
    frames: np.ndarray
    forces: np.ndarray
    origins: tuple
    response: object
    residual_norm: float
    iterations: int
    mixed_evaluations: int

class RestartError(ValueError):
    """Checkpoint rejected before exposing a restored mutable model."""


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise RestartError('exact checkpoint schema required')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RestartError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise RestartError('nonfinite JSON constant')


def _decode(value, schema):
    if schema is float:
        if type(value) is not float or not math.isfinite(value):
            raise RestartError('finite JSON float required; no coercion')
        return value
    if schema is bool:
        if type(value) is not bool:
            raise RestartError('JSON boolean required')
        return value
    kind, specification = schema
    if kind == 'integer':
        if type(value) is not int or not 0 <= value <= specification:
            raise RestartError('bounded JSON integer required')
        return value
    if kind == 'array':
        shape = specification
        def nested(item, dimensions):
            if not dimensions:
                return _decode(item, float)
            if type(item) is not list or len(item) != dimensions[0]:
                raise RestartError('exact array shape required')
            return [nested(part, dimensions[1:]) for part in item]
        made = np.array(nested(value, shape), dtype=float)
        made.setflags(write=False)
        return made
    if kind == 'sequence':
        count, item_schema = specification
        if type(value) is not list or len(value) != count:
            raise RestartError('exact station/history count required')
        return tuple(_decode(item, item_schema) for item in value)
    cls, fields = kind, specification
    _keys(value, fields)
    decoded = cls(**{key: _decode(value[key], item) for key, item in fields.items()})
    return _history(decoded) if cls is SectionHistory else decoded


def _schemas(order):
    array = lambda *shape: ('array', shape)
    integer = lambda maximum: ('integer', maximum)
    history = (SectionHistory, {'plastic_coordinate': float, 'accumulated': float})
    histories = ('sequence', (2*order, history))
    point = (SectionResponse, {'origin': history, 'history': history, 'strain': array(6),
        'resultants': array(6), 'tangent': array(6, 6), 'incremental_potential': float,
        'stored_energy': float, 'dissipation_increment': float, 'plastic_active': bool})
    station = (BeamStationTrial, {'cell': integer(1), 'index': integer(order-1),
        'reference_coordinate': float, 'response': point})
    response = (NonlinearCondensedResponse, {'potential': float, 'residual': array(18), 'tangent': array(18, 18),
        'local_rotations': array(2, 3, 3), 'moments': array(2, 2, 3),
        'stations': ('sequence', (2*order, station)), 'local_residual_norm': float,
        'iterations': integer(25), 'evaluations': integer(64)})
    state = (NonlinearPathState, {'epoch': integer(2147483647), 'positions': array(3, 3),
        'frames': array(3, 3, 3), 'forces': array(3, 3), 'histories': histories})
    trial = (NonlinearPathTrial, {'origin_epoch': integer(2147483646), 'positions': array(3, 3),
        'frames': array(3, 3, 3), 'forces': array(3, 3), 'origins': histories, 'response': response,
        'residual_norm': float, 'iterations': integer(16), 'mixed_evaluations': integer(256)})
    return state, trial
