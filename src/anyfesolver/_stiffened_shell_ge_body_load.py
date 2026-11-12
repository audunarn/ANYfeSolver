"""Model-bound uniform gravity work for the native GE point owner.

V1_DEV binds 24-point native kinetic quadrature, reference shell area mass,
objective Q4 finite pullback, and the existing stable-translation elastic S3
arithmetic. The latter uses the native-material-stable inner checkpoint.
Historical nodal/staged routes and numerical policies remain unchanged.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES
from ._stiffened_shell_staged_load_experimental import _canonical, _identity, _json, _number
from ._stiffened_shell_native_gravity import compile_native_gravity, _owned, _sum, _product
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._stiffened_shell_physical_q4_mass import physical_mass, _positive

POLICY = 'STIFFENED_SHELL_GE_UNIFORM_BODY_ACCELERATION_V1_DEV'
MAGIC = b'ANYSGE_BODY1\0'
MAX_METADATA_BYTES = 4096
MAX_COMPILED_VALUES = 1_000_000


def _admission_error(owner):
    """Keep the new footprint intersection narrower than either parent route."""
    definition = owner._definition
    if definition.distributing_footprints or definition.finite_distributing_footprints:
        return 'GE body loads exclude reference-linear distributing footprints'
    if definition.lines or owner._line_samples:
        return 'GE body load admits point owners only; lines unsupported'
    if not definition.footprints:
        return None if definition.points else 'GE body load requires point attachments'
    from ._stiffened_shell_nodal_footprint import INCREMENT_ORIGIN_CHART_POLICY
    from ._stiffened_shell_q4_algorithmic_pullback import POLICY as Q4_POLICY
    from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
    from ._stiffened_shell_ge_reference_modal import _physical_spatial_inertia
    panel = definition.shell
    if (definition.shell_reference_policy != 'flat' or not panel.q4_elements or
            panel.s3_elements or panel.j2_curve is not None or
            definition.q4_pullback_policy != Q4_POLICY or
            definition.footprint_chart_policy != INCREMENT_ORIGIN_CHART_POLICY or
            not owner._point_incidence or owner._has_physical_fibre):
        return 'GE body footprint load requires flat elastic Q4 and increment-origin charts'
    # A generalized law can yield. Candidate/restore history guards continue to
    # reject every nonelastic response; no yield-magnitude classification is used.
    try:
        _positive(panel.density_kg_m3, 'shell density')
        _positive(panel.thickness_m, 'shell thickness')
        if len(owner._reference_modal_inertias) != len(owner._beam_parts):
            return 'GE body footprint load requires explicit physical beam inertia'
        for (element, _, _), inertia in zip(owner._beam_parts, owner._reference_modal_inertias, strict=True):
            if type(element.section) is not EllipsoidalGeneralizedSection:
                return 'GE body footprint load requires declared generalized sections'
            element.section.guard()
            _physical_spatial_inertia(inertia)
            np.linalg.cholesky(inertia)
    except (ValueError, np.linalg.LinAlgError):
        return 'GE body footprint load requires unchanged generalized laws and physical mass'
    return None


def admits_owner(owner):
    return _admission_error(owner) is None


@dataclass(frozen=True)
class GEBodyLoad:
    model_identity: str
    acceleration_m_s2: tuple
    identity: str = field(init=False)

    def __post_init__(self):
        identity = _identity(self.model_identity)
        if not isinstance(self.acceleration_m_s2, (tuple, list, np.ndarray)) or np.shape(self.acceleration_m_s2) != (3,):
            raise ValueError('three finite spatial acceleration components required')
        acceleration = tuple(_number(v) for v in self.acceleration_m_s2)
        if not any(acceleration):
            raise ValueError('nonzero uniform body acceleration required')
        object.__setattr__(self, 'model_identity', identity)
        object.__setattr__(self, 'acceleration_m_s2', acceleration)
        object.__setattr__(self, 'identity', sha256(_canonical(self._core())).hexdigest())

    def _core(self):
        return dict(policy=POLICY, model_identity=self.model_identity,
                    acceleration_m_s2=list(self.acceleration_m_s2))

    def to_payload(self):
        return {**self._core(), 'identity': self.identity}

    def payload(self):
        return self.to_payload()

    def serialize(self):
        return _canonical(self.to_payload())

    @classmethod
    def from_payload(cls, value):
        if (type(value) is not dict or set(value) != {'policy', 'model_identity', 'acceleration_m_s2', 'identity'}
                or value['policy'] != POLICY or type(value['acceleration_m_s2']) is not list):
            raise ValueError('invalid GE body-load descriptor')
        made = cls(value['model_identity'], value['acceleration_m_s2'])
        if value['identity'] != made.identity or _canonical(value) != made.serialize():
            raise ValueError('GE body-load descriptor identity/canonical mismatch')
        return made

    @classmethod
    def deserialize(cls, raw):
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_METADATA_BYTES:
            raise ValueError('bounded body-load descriptor bytes required')
        made = cls.from_payload(_json(raw))
        if raw != made.serialize():
            raise ValueError('canonical body-load descriptor required')
        return made

    def compile(self, owner):
        if getattr(owner, '_identity', None) != self.model_identity:
            raise ValueError('body load belongs to a different model')
        owner._require_open()
        definition = owner._definition
        admission_error = _admission_error(owner)
        if admission_error is not None:
            raise ValueError(admission_error)
        count = owner._K.shape[0]
        if type(count) is not int or not 0 < count <= MAX_COMPILED_VALUES:
            raise ValueError('bounded body-load coordinate layout required')
        panel = definition.shell
        rho = _positive(panel.density_kg_m3, 'shell density')
        thickness = _positive(panel.thickness_m, 'shell thickness')
        shell_xyz = _owned(owner._shell_xyz)
        shell_terms = [[] for _ in shell_xyz]
        for _, _, slots, _ in owner._shell_parts:
            nodes = np.asarray(slots)[::6]//6
            vertices = shell_xyz[nodes]
            if len(nodes) == 4:
                normal = np.cross(vertices[1]-vertices[0], vertices[3]-vertices[0])
                length = np.linalg.norm(normal)
                if not np.isfinite(length) or length <= 0.:
                    raise ValueError('representable Q4 reference normal required')
                matrix = physical_mass(vertices, density_kg_m3=rho, thickness_m=thickness,
                                       reference_normal=normal/length)
                masses = np.array([_sum(matrix[6*i, ::6]) for i in range(4)])
            elif len(nodes) == 3:
                area = .5*np.linalg.norm(np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]))
                if not np.isfinite(area) or area <= 0.:
                    raise ValueError('positive representable S3 reference area required')
                try:
                    mass = float(Fraction(rho)*Fraction(thickness)*Fraction(float(area))/3)
                except OverflowError as error:
                    raise ValueError('S3 body mass is not representable') from error
                masses = np.full(3, mass)
            else:
                raise ValueError('native body load requires S3 or Q4 shell cells')
            if not np.isfinite(masses).all() or np.any(masses <= 0.):
                raise ValueError('positive representable shell nodal body masses required')
            for node, mass in zip(nodes, masses, strict=True):
                shell_terms[int(node)].append(float(mass))
        shell_masses = _owned([_sum(row) for row in shell_terms])
        if len(owner._reference_modal_inertias) != len(owner._beam_parts):
            raise ValueError('complete explicit GE section inertia required')
        beams = []
        for inertia, (element, slots, _) in zip(owner._reference_modal_inertias, owner._beam_parts, strict=True):
            kernel = compile_native_gravity(element.operator.reference, inertia)
            indices = tuple(int(v)+owner._shell_count for v in slots)
            if len(indices) != 42 or min(indices) < 0 or max(indices) >= count:
                raise ValueError('native body-load full-42 owner layout required')
            beams.append((kernel, indices))
        total_mass = _sum((*shell_masses, *(m for kernel, _ in beams for m in kernel.nodal_masses)))
        if total_mass <= 0.:
            raise ValueError('positive body mass required')
        augmented = sha256(_canonical(dict(policy=POLICY, model_identity=self.model_identity,
                                           body_load=self.identity))).hexdigest()
        context = GEBodyContext(self, augmented, count, shell_xyz, shell_masses, tuple(beams), total_mass,
                                freeze_trial_load_norm=bool(definition.footprints))
        reference = SimpleNamespace(q=np.zeros(count), cells=np.tile(np.eye(3), (len(beams), 2, 1, 1)))
        gradient, _, recovery = context.evaluate(reference)
        object.__setattr__(context, 'reference_gradient', _owned(gradient))
        object.__setattr__(context, 'reference_recovery', recovery)
        return context


@dataclass(frozen=True)
class GEBodyRecovery:
    identity: str
    acceleration_m_s2: tuple
    total_mass: float
    applied_force: tuple
    applied_moment_about_origin: tuple
    work: float

    def scaled(self, factor):
        factor = _number(factor)
        return GEBodyRecovery(self.identity, tuple(_product(self.acceleration_m_s2, factor)),
                              self.total_mass, tuple(_product(self.applied_force, factor)),
                              tuple(_product(self.applied_moment_about_origin, factor)),
                              float(_product(self.work, factor)))


@dataclass(frozen=True)
class GEBodyContext:
    request: GEBodyLoad
    augmented_model_identity: str
    count: int
    shell_xyz: np.ndarray
    shell_masses: np.ndarray
    beams: tuple
    total_mass: float
    # New footprint/body route only. Compare backtracks in their parent's work
    # norm when eccentric mass activates previously quiet torque rows. Strict
    # current-state equilibrium and compatibility acceptance remain unchanged.
    freeze_trial_load_norm: bool = False
    reference_gradient: np.ndarray = field(init=False, repr=False)
    reference_recovery: GEBodyRecovery = field(init=False)

    @property
    def result_field(self):
        return 'body_load'

    def checkpoint(self, inner):
        return encode_checkpoint(self, inner)

    def evaluate(self, state):
        q = np.asarray(state.q)
        cells = np.asarray(state.cells)
        if (np.iscomplexobj(q) or q.shape != (self.count,) or not np.isfinite(q).all()
                or cells.shape != (len(self.beams), 2, 3, 3)):
            raise ValueError('finite matching native body-load state required')
        acceleration = np.asarray(self.request.acceleration_m_s2)
        terms = [[] for _ in range(self.count)]
        rows, columns, values = [], [], []
        forces, moments, work = [], [], []
        for node, (xyz, mass) in enumerate(zip(self.shell_xyz, self.shell_masses, strict=True)):
            force = _product(mass, acceleration)
            displacement = q[6*node:6*node+3]
            for axis in range(3):
                terms[6*node+axis].append(float(force[axis]))
            work.extend(_product(force, displacement))
            forces.append(force)
            moments.extend((np.cross(xyz, force), np.cross(displacement, force)))
        for cell, (kernel, indices) in zip(cells, self.beams, strict=True):
            displacement = q[list(indices[:18])].reshape(3, 6)[:, :3]
            high, low = np.empty((3, 3)), np.empty((3, 3))
            for at in np.ndindex(high.shape):
                high[at], low[at] = split_sum((float(kernel.coordinates[at]), float(displacement[at])))
            result = kernel.evaluate(high, low, cell, acceleration)
            for slot, value in zip(indices, result.gradient, strict=True):
                terms[slot].append(float(value))
            rr, cc = np.nonzero(result.spatial_jacobian)
            rows.extend(indices[int(r)] for r in rr)
            columns.extend(indices[int(c)] for c in cc)
            values.extend(result.spatial_jacobian[rr, cc])
            forces.append(result.total_force)
            moments.append(result.total_moment)
            work.append(result.value)
        gradient = _owned([_sum(row) for row in terms])
        jacobian = sparse.coo_matrix((values, (rows, columns)), shape=(self.count, self.count)).tocsr()
        if not np.isfinite(jacobian.data).all():
            raise ValueError('representable sparse body-load Jacobian required')
        recovery = GEBodyRecovery(self.request.identity, self.request.acceleration_m_s2, self.total_mass,
                                  tuple(_sum(v[i] for v in forces) for i in range(3)),
                                  tuple(_sum(v[i] for v in moments) for i in range(3)), _sum(work))
        return gradient, jacobian, recovery


def encode_checkpoint(context, inner):
    if type(context) is not GEBodyContext:
        raise TypeError('compiled GE body context required')
    if type(inner) is not bytes or not inner or len(inner) > MAX_CHECKPOINT_BYTES:
        raise ValueError('bounded body-load inner checkpoint required')
    header = _canonical(dict(policy=POLICY, request=context.request.to_payload(),
                             inner_sha256=sha256(inner).hexdigest()))
    if len(header) > MAX_METADATA_BYTES or len(MAGIC)+4+len(header)+len(inner)+32 > MAX_CHECKPOINT_BYTES:
        raise ValueError('body-load checkpoint byte bound exceeded')
    content = MAGIC+len(header).to_bytes(4, 'little')+header+inner
    return content+sha256(content).digest()


def decode_checkpoint(raw):
    if (type(raw) is not bytes or not raw.startswith(MAGIC)
            or not len(MAGIC)+4+32 < len(raw) <= MAX_CHECKPOINT_BYTES):
        raise ValueError('invalid bounded body-load checkpoint framing')
    start = len(MAGIC)+4
    size = int.from_bytes(raw[len(MAGIC):start], 'little')
    if not 0 < size <= MAX_METADATA_BYTES or start+size >= len(raw)-32:
        raise ValueError('invalid body-load checkpoint metadata bound')
    if sha256(memoryview(raw)[:-32]).digest() != raw[-32:]:
        raise ValueError('body-load checkpoint digest mismatch')
    header_raw = raw[start:start+size]
    header = _json(header_raw)
    if (type(header) is not dict or set(header) != {'policy', 'request', 'inner_sha256'}
            or header['policy'] != POLICY or _canonical(header) != header_raw):
        raise ValueError('invalid canonical body-load checkpoint metadata')
    request = GEBodyLoad.from_payload(header['request'])
    inner = raw[start+size:-32]
    if header['inner_sha256'] != sha256(inner).hexdigest():
        raise ValueError('body-load inner checkpoint digest mismatch')
    return request, inner
