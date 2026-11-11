"""Reconstructible native curved-beam definitions, not activation or restart.

Only two explicit existing candidate classes are admitted. No class import is
selected by input data. A definition creates a fresh element; accepted history
must still enter through its existing externally authenticated owner protocol.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite

import numpy as np

from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from ._ge_beam3_fibre_section import PhysicalFibreSection, Fibre, FlowCurve, POLICY as FIBRE_POLICY
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection, POLICY as GENERALIZED_POLICY
from ._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_p5_seeded.core import canonical

SCHEMA = 'GE_BEAM3_NATIVE_RECONSTRUCTIBLE_DEFINITION_V1'
PRECISE_SCHEMA = 'GE_BEAM3_NATIVE_RECONSTRUCTIBLE_DEFINITION_PRECISE_V2'
STABLE_SO3_SCHEMA = 'GE_BEAM3_NATIVE_RECONSTRUCTIBLE_DEFINITION_STABLE_SO3_V3'
COMPENSATED_SCHEMA = 'GE_BEAM3_NATIVE_RECONSTRUCTIBLE_DEFINITION_COMPENSATED_STATE_V4_DEV'
STATIC_POLICY = 'EIGHTEEN_EXTERNAL_TWENTY_FOUR_INTERNAL_STATIC_STATIONARITY'
DYNAMIC_POLICY = 'RETAIN_PHYSICAL_CELL_INERTIA_NO_STATIC_MASS_SUBSTITUTION'
MAX_BYTES = 2 * 1024 * 1024
KEYS = frozenset(('schema', 'family', 'formulation_id', 'element_id', 'node_ids',
                  'reference', 'section', 'section_inertia', 'quadrature',
                  'element_identity', 'operator_identity', 'section_identity',
                  'static_policy', 'dynamic_policy', 'production_qualified'))
REFERENCE_KEYS = frozenset(('coordinates', 'nodal_triads', 'evaluation_id',
                            'regularity_relative_tolerance', 'rotation_tolerance', 'frame_tolerance'))


class DefinitionError(ValueError):
    """The record cannot reconstruct this exact native candidate definition."""


def _require(ok, message):
    if not ok:
        raise DefinitionError(message)


def _keys(value, keys):
    _require(type(value) is dict and set(value) == set(keys), 'exact native definition schema required')


def _definition_keys(value):
    _require(type(value) is dict, 'exact native definition object required')
    if value.get('schema') == COMPENSATED_SCHEMA:
        from ._stiffened_shell_f06_increment_arithmetic import POLICY
        _keys(value, KEYS | {'arithmetic_policy'})
        _require(value['family'] in ('RESULTANT_ELLIPSOID', 'PHYSICAL_AXIAL_BIAXIAL_FIBRE') and value['arithmetic_policy'] == POLICY,
                 'exact native compensated state arithmetic policy required')
    elif value.get('schema') == STABLE_SO3_SCHEMA:
        from ._ge_beam3_centered_so3_strain import POLICY
        _keys(value, KEYS | {'arithmetic_policy'})
        _require(value['family'] == 'RESULTANT_ELLIPSOID' and value['arithmetic_policy'] == POLICY,
                 'exact generalized stable SO3 arithmetic policy required')
    elif value.get('schema') == PRECISE_SCHEMA:
        from ._ge_beam3_precise_geometric_work import POLICY
        _keys(value, KEYS | {'arithmetic_policy'})
        _require(value['family'] == 'RESULTANT_ELLIPSOID' and value['arithmetic_policy'] == POLICY,
                 'exact generalized precise arithmetic policy required')
    else:
        _keys(value, KEYS)
        _require(value['schema'] == SCHEMA, 'native definition schema mismatch')


def _load(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES, 'bounded definition bytes required')

    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate definition key')
            result[key] = value
        return result

    def number(text):
        value = float(text)
        _require(isfinite(value), 'finite definition numbers required')
        return value

    def forbidden(text):
        raise DefinitionError('nonfinite definition token: '+text)

    try:
        value = json.loads(raw.decode('ascii'), object_pairs_hook=pairs,
                           parse_float=number, parse_constant=forbidden)
        _require(canonical(value) == raw, 'canonical definition bytes required')
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise DefinitionError('invalid bounded definition JSON') from error
    _definition_keys(value)
    return value


def _array(value, shape):
    # Do not coerce booleans, strings, ragged rows or exotic user objects.
    def scalars(v):
        if type(v) is list:
            for item in v:
                scalars(item)
        else:
            _require(type(v) in (int, float) and isfinite(v), 'finite numeric definition array required')
    scalars(value)
    try:
        result = np.array(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise DefinitionError('invalid definition array') from error
    _require(result.shape == shape and np.isfinite(result).all(), 'definition array shape/range')
    return np.frombuffer(result.tobytes(), dtype=float).reshape(shape)


def _inertia(value):
    made = _array(value, (6, 6))
    _require(np.array_equal(made, made.T), 'exact symmetric physical section inertia required')
    np.linalg.cholesky(made)
    return made


def _describe(element, inertia):
    _require(type(element) in (NativeGeneralizedStaticElement, NativeFibreStaticElement),
             'exact native generalized or physical-fibre element required')
    core = element.operator
    core.guard()
    _require(element.section is core.section and element.production_qualified is False,
             'native definition section/qualification authority')
    ref = core.reference
    reference = dict(coordinates=ref.coordinates, nodal_triads=ref.nodal_triads,
                     evaluation_id=ref.evaluation_id,
                     regularity_relative_tolerance=ref._regularity_relative_tolerance,
                     rotation_tolerance=ref._rotation_tolerance, frame_tolerance=ref._frame_tolerance)
    if type(element) is NativeGeneralizedStaticElement:
        _require(type(core.section) is EllipsoidalGeneralizedSection, 'generalized section family mismatch')
        section = dict(policy=GENERALIZED_POLICY, elastic=core.section.elastic,
                       metric=core.section.metric, yield_force=core.section.yield_force,
                       hardening=core.section.hardening)
        family = 'RESULTANT_ELLIPSOID'
    else:
        _require(type(core.section) is PhysicalFibreSection, 'physical fibre family mismatch')
        section = dict(policy=FIBRE_POLICY, fibres=[asdict(f) for f in core.section.fibres],
                       background_factor=core.section.background_factor)
        family = 'PHYSICAL_AXIAL_BIAXIAL_FIBRE'
    result = dict(schema=SCHEMA, family=family, formulation_id=element.formulation_id,
                element_id=element.element_id, node_ids=list(element.node_ids), reference=reference,
                section=section, section_inertia=inertia, quadrature=core.order,
                element_identity=element.identity, operator_identity=core.identity,
                section_identity=core.section.identity, static_policy=STATIC_POLICY,
                dynamic_policy=DYNAMIC_POLICY, production_qualified=False)
    if core.arithmetic_policy is not None:
        from ._ge_beam3_centered_so3_strain import POLICY as STABLE_SO3_POLICY
        from ._stiffened_shell_f06_increment_arithmetic import POLICY as COMPENSATED_POLICY
        result.update(schema=COMPENSATED_SCHEMA if core.arithmetic_policy == COMPENSATED_POLICY else
                      STABLE_SO3_SCHEMA if core.arithmetic_policy == STABLE_SO3_POLICY else PRECISE_SCHEMA,
                      arithmetic_policy=core.arithmetic_policy)
    return result


def _build(data):
    _definition_keys(data)
    _require(data['production_qualified'] is False
             and data['static_policy'] == STATIC_POLICY and data['dynamic_policy'] == DYNAMIC_POLICY,
             'native definition scope/policy mismatch')
    ids = data['node_ids']
    _require(type(data['element_id']) is int and data['element_id'] > 0
             and type(ids) is list and len(ids) == 3 and all(type(i) is int and i > 0 for i in ids)
             and len(set(ids)) == 3, 'three distinct native nodes and element ID required')
    _require(type(data['quadrature']) is int and data['quadrature'] in (4, 8), 'explicit native quadrature required')
    r = data['reference']; _keys(r, REFERENCE_KEYS)
    for name in ('regularity_relative_tolerance', 'rotation_tolerance', 'frame_tolerance'):
        _require(type(r[name]) is float and isfinite(r[name]) and r[name] > 0., 'explicit reference tolerance required')
    ref = Reference(_array(r['coordinates'], (3, 3)), _array(r['nodal_triads'], (3, 3, 3)),
                    regularity_relative_tolerance=r['regularity_relative_tolerance'],
                    rotation_tolerance=r['rotation_tolerance'], frame_tolerance=r['frame_tolerance'])
    _require(ref.evaluation_id == r['evaluation_id'], 'reference evaluation identity changed')
    section = data['section']
    if data['family'] == 'RESULTANT_ELLIPSOID':
        _keys(section, ('policy', 'elastic', 'metric', 'yield_force', 'hardening'))
        _require(section['policy'] == GENERALIZED_POLICY, 'generalized law authority mismatch')
        law = EllipsoidalGeneralizedSection(_array(section['elastic'], (6, 6)),
                _array(section['metric'], (6, 6)), section['yield_force'], section['hardening'])
        cls = NativeGeneralizedStaticElement
    elif data['family'] == 'PHYSICAL_AXIAL_BIAXIAL_FIBRE':
        _keys(section, ('policy', 'fibres', 'background_factor'))
        _require(section['policy'] == FIBRE_POLICY and type(section['fibres']) is list
                 and 1 <= len(section['fibres']) <= 1024, 'physical fibre law authority mismatch')
        fibres = []
        for row in section['fibres']:
            _keys(row, ('fibre_id', 'y', 'z', 'area', 'young', 'curve'))
            c = row['curve']; _keys(c, ('plastic_strain', 'flow_stress', 'tail_slope'))
            _require(type(c['plastic_strain']) is list and type(c['flow_stress']) is list,
                     'explicit native flow-curve arrays required')
            curve = FlowCurve(tuple(c['plastic_strain']), tuple(c['flow_stress']), c['tail_slope'])
            fibres.append(Fibre(row['fibre_id'], row['y'], row['z'], row['area'], row['young'], curve))
        factor = section['background_factor']
        _require(type(factor) is list and 1 <= len(factor) <= 64, 'bounded physical background factor')
        law = PhysicalFibreSection(tuple(fibres), _array(factor, (len(factor), 6)))
        cls = NativeFibreStaticElement
    else:
        raise DefinitionError('unknown native beam section family; no legacy fallback')
    inertia = _inertia(data['section_inertia'])
    extra = {'arithmetic_policy': data['arithmetic_policy']} if data['schema'] in (PRECISE_SCHEMA, STABLE_SO3_SCHEMA, COMPENSATED_SCHEMA) else {}
    element = cls(data['element_id'], tuple(ids), ref, law, order=data['quadrature'], **extra)
    # Reconstruct all derived identities. Hashes alone never substitute for data.
    _require(canonical(_describe(element, inertia)) == canonical(data),
             'reconstructed native definition/identity differs')
    return element, inertia


@dataclass(frozen=True)
class NativeBeamDefinition:
    """Immutable definition bytes; creates fresh unbound elements, never state."""
    raw: bytes

    def __post_init__(self):
        _build(_load(self.raw))

    @property
    def sha256(self):
        return sha256(self.raw).hexdigest()

    @classmethod
    def capture(cls, element, section_inertia):
        inertia = _inertia(np.asarray(section_inertia).tolist())
        return cls(canonical(_describe(element, inertia)))

    @classmethod
    def from_bytes(cls, raw, *, expected_sha256):
        _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES
                 and type(expected_sha256) is str
                 and sha256(raw).hexdigest() == expected_sha256, 'external definition hash mismatch')
        return cls(raw)

    def instantiate(self):
        """Return a fresh existing element and read-only physical inertia tensor."""
        return _build(_load(self.raw))
