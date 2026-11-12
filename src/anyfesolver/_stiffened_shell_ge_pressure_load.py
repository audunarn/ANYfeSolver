"""Owned shell interpolation-surface pressure, using existing native kernels.

Reference-dead V2D S3 loads retain enhanced rotational covectors and are admitted
only for reference-linear work. Current-surface loads act on translations, with
the existing generally nonsymmetric spatial Jacobian and no invented potential.
The Q4 current-surface admission rejects degenerate quadrature stations instead
of inheriting the old generic kernel's silent skip. No legacy kernel is changed.
An explicitly selected director-enhanced S3 dead surface is a separate elastic
development law, not a reinterpretation of the native reference-dead vector.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .boundary import LoadCase, _SMALL
from .e4_pl_element import QualifiedE4PLShellElement as Q4
from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement as S3
from ._stiffened_shell_staged_load_experimental import _canonical, _identity, _number, _json
from ._stiffened_shell_ge_checkpoint_binary import MAX_CHECKPOINT_BYTES
from ._stiffened_shell_native_gravity import _owned, _sum, _product, _finite_array

POLICY = 'STIFFENED_SHELL_GE_NATIVE_SHELL_PRESSURE_LOAD_V1_DEV'
DIRECTOR_CONFIGURATION = 'reference_dead_director_v2'
DIRECTOR_LOAD_POLICY = 'STIFFENED_SHELL_GE_DIRECTOR_PRESSURE_LOAD_V2_DEV'
VECTOR_EQUILIBRIUM_POLICY = 'STIFFENED_SHELL_PRESSURE_VECTOR_EQUILIBRIUM_V1_DEV'
MAGIC = b'ANYSGE_PRESS1\0'
MAX_METADATA_BYTES = 1024*1024
MAX_ACTIONS = 16384


@dataclass(frozen=True)
class GEShellPressureAction:
    element_id: int
    pressure_pa: float
    configuration: str = 'reference_dead'

    def __post_init__(self):
        if type(self.element_id) is not int or self.element_id <= 0:
            raise ValueError('positive shell element identifier required')
        object.__setattr__(self, 'pressure_pa', _number(self.pressure_pa))
        if self.pressure_pa == 0.:
            raise ValueError('nonzero signed shell pressure required')
        if type(self.configuration) is not str or self.configuration not in ('reference_dead', 'current_surface', DIRECTOR_CONFIGURATION):
            raise ValueError('explicit admitted shell pressure configuration required')

    def payload(self):
        return dict(element_id=self.element_id, pressure_pa=self.pressure_pa, configuration=self.configuration)


@dataclass(frozen=True)
class GEShellPressureLoad:
    """Owned pressure request with optional, checkpoint-bound finite assessment.

    ``equilibrium_policy=None`` preserves component-relative checks. The
    vector policy checks each shell node's free force and moment vector, plus
    global work; a small component at a node may be masked by a larger one.
    """
    model_identity: str
    actions: tuple[GEShellPressureAction,...]
    equilibrium_policy: str | None = None
    identity: str=field(init=False)

    def __post_init__(self):
        object.__setattr__(self,'model_identity',_identity(self.model_identity))
        if type(self.actions) is not tuple or not 0<len(self.actions)<=MAX_ACTIONS or any(type(a) is not GEShellPressureAction for a in self.actions):
            raise ValueError('bounded tuple of shell pressure actions required')
        if (self.equilibrium_policy is not None and
                (type(self.equilibrium_policy) is not str or
                 self.equilibrium_policy != VECTOR_EQUILIBRIUM_POLICY)):
            raise ValueError('explicit admitted pressure equilibrium policy required')
        ids=[a.element_id for a in self.actions]
        if len(set(ids))!=len(ids):raise ValueError('duplicate shell pressure action element')
        object.__setattr__(self,'actions',tuple(sorted(self.actions,key=lambda a:a.element_id)))
        object.__setattr__(self,'identity',sha256(_canonical(self._core())).hexdigest())
        if len(self.serialize())>MAX_METADATA_BYTES:raise ValueError('bounded pressure descriptor required')

    @property
    def policy(self):
        return (DIRECTOR_LOAD_POLICY if any(a.configuration == DIRECTOR_CONFIGURATION for a in self.actions)
                else POLICY)

    def _core(self):
        core = dict(policy=self.policy,model_identity=self.model_identity,actions=[a.payload() for a in self.actions])
        if self.equilibrium_policy is not None:
            core['equilibrium_policy'] = self.equilibrium_policy
        return core

    def to_payload(self):return dict(self._core(),identity=self.identity)
    def serialize(self):return _canonical(self.to_payload())

    @classmethod
    def from_payload(cls,value):
        if (type(value) is not dict or
                set(value) not in ({'policy','model_identity','actions','identity'},
                                   {'policy','model_identity','actions','identity','equilibrium_policy'}) or
                value['policy'] not in (POLICY, DIRECTOR_LOAD_POLICY) or type(value['actions']) is not list):
            raise ValueError('strict pressure load descriptor required')
        if not 0<len(value['actions'])<=MAX_ACTIONS:raise ValueError('bounded action inventory required')
        rows=[]
        for row in value['actions']:
            if type(row) is not dict or set(row)!={'element_id','pressure_pa','configuration'}:
                raise ValueError('strict shell pressure action required')
            rows.append(GEShellPressureAction(**row))
        result=cls(value['model_identity'],tuple(rows), value.get('equilibrium_policy'))
        if result.identity!=value['identity'] or result.serialize()!=_canonical(value):
            raise ValueError('pressure descriptor identity/canonical mismatch')
        return result

    @classmethod
    def deserialize(cls,raw):
        if type(raw) is not bytes or not 0<len(raw)<=MAX_METADATA_BYTES:
            raise ValueError('bounded pressure load bytes required')
        result=cls.from_payload(_json(raw))
        if result.serialize()!=raw:raise ValueError('canonical pressure load bytes required')
        return result

    def compile(self, owner):
        owner._require_open()
        if owner._definition.distributing_footprints or owner._definition.finite_distributing_footprints:
            raise ValueError('pressure loads exclude reference-linear distributing footprints')
        if self.model_identity != owner._identity:
            raise ValueError('pressure load belongs to a different model')
        if owner._line_samples or owner._definition.lines or owner._definition.footprints or not owner._definition.points:
            raise ValueError('pressure load currently requires point-owner attachments')
        parts = {eid: (element, slots) for eid, element, slots, _ in owner._shell_parts}
        if len(parts) != len(owner._shell_parts):
            raise ValueError('unique owner shell element mapping required')
        mesh = owner._shell_model.mesh
        count = owner._K.shape[0]
        shell_xyz = _finite_array(owner._shell_xyz, (owner._shell_count//6, 3), 'owned shell reference coordinates')
        patches = []
        for action in self.actions:
            if action.element_id not in parts:
                raise ValueError('pressure action references unknown shell element')
            element, raw_slots = parts[action.element_id]
            if type(element) not in (Q4, S3) or element.element_id != action.element_id:
                raise ValueError('exact owner-bound Q4 or native V2D S3 shell required')
            if action.configuration == DIRECTOR_CONFIGURATION:
                if type(element) is not S3:
                    raise ValueError('director-enhanced pressure requires native S3')
                if owner._definition.shell.j2_curve is not None:
                    raise ValueError('director-enhanced pressure currently requires virgin elastic shells')
            slots = np.asarray(raw_slots)
            nodes = len(element.node_ids)
            if (slots.shape != (6*nodes,) or not np.issubdtype(slots.dtype, np.integer)
                    or np.any(slots < 0) or np.any(slots >= owner._shell_count)
                    or not np.array_equal(slots.reshape(nodes, 6), slots[::6, None]+np.arange(6))
                    or np.any(slots[::6] % 6) or len(set(slots.tolist())) != len(slots)):
                raise ValueError('strict shell pressure coordinate mapping required')
            xyz = _owned(shell_xyz[slots[::6]//6])
            binding = _binding(element, mesh)
            native_xyz = _coordinates(element, mesh)
            if not np.array_equal(xyz, native_xyz):
                raise ValueError('pressure shell reference mapping mismatch')
            load = LoadCase('GE shell pressure native kernel')
            dead = None
            if action.configuration in ('reference_dead', DIRECTOR_CONFIGURATION):
                dead = _owned(_finite_array(load._consistent_pressure_load(element, mesh, action.pressure_pa),
                                           (6*nodes,), 'native reference pressure covector'))
            patch = _PressurePatch(action, element, mesh, tuple(int(v) for v in slots), xyz, binding, dead)
            patches.append(patch)
        identity = sha256(_canonical(dict(model=self.model_identity, pressure_load=self.identity))).hexdigest()
        context = GEPressureContext(self, identity, count, tuple(patches))
        force, _, recovery = context.evaluate(SimpleNamespace(q=np.zeros(count)))
        object.__setattr__(context, 'reference_gradient', force)
        object.__setattr__(context, 'reference_recovery', recovery)
        return context


def _coordinates(element, mesh):
    values = element._coordinates(mesh) if type(element) is S3 else element.get_node_coordinates(mesh)
    return _finite_array(values, (len(element.node_ids), 3), 'native pressure reference coordinates')


def _binding(element, mesh):
    """Bounded local native input guard, with no full-model search or hashing."""
    if type(element) not in (Q4, S3):
        raise ValueError('exact native pressure shell required')
    if mesh.elements.get(element.element_id) is not element:
        raise ValueError('captured native pressure element registry binding changed')
    normal = tuple(np.asarray(getattr(element, 'reference_normal', ()), dtype=float).ravel())
    rule = ()
    if type(element) is Q4:
        rule = (np.asarray(element.gauss_points, dtype=float).tobytes(),
                np.asarray(element.gauss_weights, dtype=float).tobytes())
    return (type(element), element.element_id, tuple(element.node_ids),
            _coordinates(element, mesh).tobytes(), normal, rule)


def _guard_current_surface(element, coordinates):
    if not np.isfinite(coordinates).all():
        raise ValueError('finite current pressure surface coordinates required')
    if type(element) is Q4:
        # The reused generic Q4 kernel silently skips sub-threshold stations.
        # This new context explicitly rejects such current surfaces instead.
        for xi, eta in element.gauss_points:
            _, dxi, deta = element.compute_shape_functions(float(xi), float(eta))
            with np.errstate(over='ignore', invalid='ignore'):
                area = np.cross(coordinates.T@dxi, coordinates.T@deta)
                norm = float(np.linalg.norm(area))
            if not np.isfinite(norm) or norm < _SMALL:
                raise ValueError('Q4 current pressure surface is degenerate or unrepresentable')
    # V2D S3's reused native load/tangent already reject degenerate surfaces.


@dataclass(frozen=True)
class _PressurePatch:
    action: GEShellPressureAction
    element: object = field(repr=False)
    mesh: object = field(repr=False)
    slots: tuple
    coordinates: np.ndarray
    binding: tuple = field(repr=False)
    dead_force: np.ndarray | None = field(repr=False)

    def evaluate(self, q, anchors=None):
        if _binding(self.element, self.mesh) != self.binding:
            raise ValueError('captured native pressure reference binding changed')
        local = q[list(self.slots)].reshape(-1, 6)
        with np.errstate(over='ignore', invalid='ignore'):
            positions = self.coordinates+local[:, :3]
        if not np.isfinite(positions).all():
            raise ValueError('finite current pressure coordinates required')
        if self.action.configuration == DIRECTOR_CONFIGURATION:
            from ._stiffened_shell_s3_dead_work_candidate import evaluate
            work, _, _, force, tangent = evaluate(self.coordinates,
                np.asarray(self.element.reference_normal), self.dead_force,
                local.ravel(), anchors=anchors, director_only=True)
            return force, tangent, positions, work
        if self.dead_force is not None:
            potential = (float(_sum(_product(self.dead_force.reshape(-1, 6)[:, :3], local[:, :3]).flat))
                         if type(self.element) is Q4 else None)
            return self.dead_force, None, positions, potential
        _guard_current_surface(self.element, positions)
        load = LoadCase('GE shell pressure native kernel')
        force = _finite_array(load._consistent_pressure_load(self.element, self.mesh, self.action.pressure_pa, positions),
                              (len(self.slots),), 'current pressure force')
        tangent = _finite_array(load._consistent_pressure_tangent(self.element, self.mesh, self.action.pressure_pa, positions),
                                (len(self.slots), len(self.slots)), 'current pressure tangent')
        return force, tangent, positions, None


@dataclass(frozen=True)
class GEPressureRecovery:
    identity: str
    applied_force: tuple
    applied_moment_about_origin: tuple
    potential: float | None
    element_ids: tuple
    moment_scope: str = 'spatial_interpolation_surface'

    def scaled(self, factor):
        factor = _number(factor)
        return GEPressureRecovery(self.identity, tuple(_product(self.applied_force, factor)),
            tuple(_product(self.applied_moment_about_origin, factor)),
            None if self.potential is None else float(_product(self.potential, factor)),
            self.element_ids, self.moment_scope)


@dataclass(frozen=True)
class GEPressureContext:
    request: GEShellPressureLoad
    augmented_model_identity: str
    count: int
    patches: tuple
    reference_gradient: np.ndarray = field(init=False, repr=False)
    reference_recovery: GEPressureRecovery = field(init=False)

    @property
    def result_field(self):
        return 'pressure_load'

    @property
    def pressure_vector_equilibrium(self):
        return self.request.equilibrium_policy == VECTOR_EQUILIBRIUM_POLICY

    @property
    def freeze_trial_load_norm(self):
        """Compare backtracks in the parent's norm; equilibrium stays unchanged."""
        return any(p.action.configuration in ('current_surface', DIRECTOR_CONFIGURATION) for p in self.patches)

    @property
    def finite_supported(self):
        return not any(type(p.element) is S3 and p.action.configuration == 'reference_dead' for p in self.patches)

    @property
    def finite_error(self):
        return '' if self.finite_supported else (
            'S3 reference_dead pressure enhanced rotational covectors support reference-linear work only; '
            'finite enhanced-dead extension is not implemented; current_surface is a distinct loading law')

    def checkpoint(self, inner):
        return encode_checkpoint(self, inner)

    def evaluate(self, state):
        q = _finite_array(state.q, (self.count,), 'pressure state')
        terms = [[] for _ in range(self.count)]
        rows, columns, values = [], [], []
        forces, moments, potentials = [], [], []
        conservative = self.finite_supported and not any(
            p.action.configuration == 'current_surface' for p in self.patches)
        for patch in self.patches:
            anchors = getattr(state, 'shell_anchors', None)
            local_anchors = None if anchors is None else anchors[np.asarray(patch.slots)[::6]//6]
            force, tangent, positions, work = patch.evaluate(q, local_anchors)
            for slot, value in zip(patch.slots, force, strict=True):
                terms[slot].append(float(value))
            if tangent is not None:
                rr, cc = np.nonzero(tangent)
                rows.extend(patch.slots[int(r)] for r in rr)
                columns.extend(patch.slots[int(c)] for c in cc)
                values.extend(tangent[rr, cc])
            nodal = force.reshape(-1, 6)
            forces.extend(nodal[:, :3])
            with np.errstate(over='ignore', invalid='ignore'):
                moments.extend(np.cross(positions, nodal[:, :3]))
            moments.extend(nodal[:, 3:])
            if work is not None:
                potentials.append(work)
        force = _owned([_sum(t) for t in terms])
        jacobian = sparse.coo_matrix((values, (rows, columns)), shape=(self.count, self.count)).tocsr()
        if not np.isfinite(jacobian.data).all():
            raise ValueError('finite pressure Jacobian required')
        moment_scope = 'spatial_interpolation_surface'
        if self.request.policy == DIRECTOR_LOAD_POLICY:
            moment_scope = ('spatial_mixed_pressure_surfaces' if any(
                p.action.configuration == 'current_surface' for p in self.patches)
                else 'spatial_director_enhanced_surface')
        if not self.finite_supported:
            moment_scope = 'reference_linear_covectors'
        recovery = GEPressureRecovery(self.request.identity,
            tuple(_sum(f[i] for f in forces) for i in range(3)),
            tuple(_sum(m[i] for m in moments) for i in range(3)),
            _sum(potentials) if conservative else None,
            tuple(a.element_id for a in self.request.actions),
            moment_scope)
        return force, jacobian, recovery


def encode_checkpoint(context,inner):
    if type(context) is not GEPressureContext or type(inner) is not bytes or not inner:
        raise ValueError('compiled pressure context and checkpoint required')
    header=_canonical(dict(policy=context.request.policy,request=context.request.to_payload(),inner_sha256=sha256(inner).hexdigest()))
    if len(header)>MAX_METADATA_BYTES or len(MAGIC)+4+len(header)+len(inner)+32>MAX_CHECKPOINT_BYTES:
        raise ValueError('pressure checkpoint byte bound exceeded')
    content=MAGIC+len(header).to_bytes(4,'little')+header+inner
    return content+sha256(content).digest()


def decode_checkpoint(raw):
    if type(raw) is not bytes or not raw.startswith(MAGIC) or not len(MAGIC)+4+32<len(raw)<=MAX_CHECKPOINT_BYTES:
        raise ValueError('bounded pressure checkpoint framing required')
    start=len(MAGIC)+4;size=int.from_bytes(raw[len(MAGIC):start],'little')
    if not 0<size<=MAX_METADATA_BYTES or start+size>=len(raw)-32 or sha256(memoryview(raw)[:-32]).digest()!=raw[-32:]:
        raise ValueError('pressure checkpoint bound/digest mismatch')
    header_raw=raw[start:start+size];header=_json(header_raw)
    if type(header) is not dict or set(header)!={'policy','request','inner_sha256'} or header['policy'] not in (POLICY, DIRECTOR_LOAD_POLICY) or _canonical(header)!=header_raw:
        raise ValueError('strict pressure checkpoint metadata required')
    request=GEShellPressureLoad.from_payload(header['request']);inner=raw[start+size:-32]
    if header['policy'] != request.policy:
        raise ValueError('pressure checkpoint work-law policy mismatch')
    if sha256(inner).hexdigest()!=header['inner_sha256']:raise ValueError('pressure inner checkpoint mismatch')
    return request,inner
